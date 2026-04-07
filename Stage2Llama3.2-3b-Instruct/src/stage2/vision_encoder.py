"""Frozen EfficientNet stream encoder for stage-2 training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose

from src.vision_modules.utils import (
    ConvertBGR2RGB,
    CropToRectangle,
    Permute,
    RescalePixelValues,
    Reshape,
    Resize,
)
from src.vision_modules.vision_model import MeanModule, load_hypermodel_weights
from src.vision_modules.sense_backbone import StridedInflatedEfficientNet
from src.stage2.runtime import resolve_runtime


def _downscale(frame: np.ndarray, max_length: int) -> np.ndarray:
    """Resize frame if one side exceeds max_length while preserving aspect ratio."""
    height, width, _ = frame.shape
    ratio = max_length / max(height, width)
    if ratio < 1:
        target_size = (int(width * ratio), int(height * ratio))
        frame = cv2.resize(frame, target_size)
    return frame


def _resize_for_vertical_pipeline(frame: np.ndarray) -> np.ndarray:
    """Mirror the extractor's conservative resize path."""
    height, width, _ = frame.shape
    return _downscale(frame, max(height, width))


def _rotate_and_square_pad(frame: np.ndarray) -> np.ndarray:
    """Rotate a landscape frame and square-pad to match the released extractor."""
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = frame.shape
    square_size = max(height, width)
    pad_top = int((square_size - height) / 2)
    pad_bottom = square_size - height - pad_top
    pad_left = int((square_size - width) / 2)
    pad_right = square_size - width - pad_left
    return cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
    )


class FrozenEfficientNetStreamEncoder(nn.Module):
    """Stateful stream encoder that reproduces the released feature extractor behavior."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float32,
        fps_out: int = 16,
        num_frames_required: int = 4,
        warmup_frames: int = 12,
    ) -> None:
        super().__init__()
        runtime = resolve_runtime(preferred_device=device, vision_dtype=torch_dtype)
        self.checkpoint_path = str(checkpoint_path)
        self.device = runtime.device
        self.torch_dtype = runtime.vision_dtype
        self.fps_out = fps_out
        self.num_frames_required = num_frames_required
        self.warmup_frames = warmup_frames
        self.spatial_res = [5, 7]

        self.transforms = Compose(
            [
                ConvertBGR2RGB(),
                CropToRectangle(aspect_ratio=1.4),
                Resize(height=224, width=160, keep_aspect_ratio=False),
                RescalePixelValues(scale=255.0),
                Permute([2, 0, 1]),
                Reshape([1, 3, 224, 160]),
            ]
        )

        self.net = nn.Sequential(
            StridedInflatedEfficientNet(),
            MeanModule(),
            nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 3031)),
        )
        load_hypermodel_weights(self.net, self.checkpoint_path, strict=False)
        self.net.to(self.device, dtype=self.torch_dtype)
        self.net.eval()
        for parameter in self.net.parameters():
            parameter.requires_grad = False

        self._captured_features: torch.Tensor | None = None
        self._frame_buffer: list[np.ndarray] = []
        self.net[0].cnn[31][0].register_forward_hook(self._capture_last_conv)

    def _capture_last_conv(self, _, __, output: torch.Tensor) -> None:
        self._captured_features = output.detach()

    def reset_stream_state(self) -> None:
        """Reset internal temporal state on steppable convolutions."""
        self._frame_buffer = []
        for module in self.net.modules():
            reset_fn = getattr(module, "reset", None)
            if callable(reset_fn):
                reset_fn()

    def _iter_sampled_frames(
        self,
        video_path: str | Path,
        rotate_90_cw: bool,
    ) -> Iterable[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps is None or source_fps <= 0:
            source_fps = float(self.fps_out)

        next_frame_timestamp = 0.0
        frame_index = -1
        try:
            while True:
                success = capture.grab()
                if not success:
                    break

                frame_index += 1
                timestamp_sec = frame_index / source_fps
                if timestamp_sec < next_frame_timestamp:
                    continue

                success, frame = capture.retrieve()
                if not success:
                    break

                frame = _resize_for_vertical_pipeline(frame)
                if rotate_90_cw:
                    frame = _rotate_and_square_pad(frame)

                next_frame_timestamp += 1.0 / self.fps_out
                yield frame
        finally:
            capture.release()

    def _consume_frame(self, frame: np.ndarray, capture_output: bool) -> torch.Tensor | None:
        self._frame_buffer.append(self.transforms(frame))
        if len(self._frame_buffer) < self.num_frames_required:
            return None

        step_input = np.concatenate(self._frame_buffer, axis=0)
        self._frame_buffer = []

        step_input_t = torch.from_numpy(step_input).to(self.device, dtype=self.torch_dtype)
        with torch.no_grad():
            self._captured_features = None
            _ = self.net(step_input_t)
            if not capture_output:
                return None
            if self._captured_features is None:
                raise RuntimeError("EfficientNet hook did not capture last-conv features")
            return self._captured_features.detach().to(torch.float32)

    @staticmethod
    def _format_feature_maps(feature_maps: torch.Tensor) -> torch.Tensor:
        """Convert [L, C, H, W] into [L, H*W, C] for the xattn adapter."""
        return feature_maps.flatten(2).permute(0, 2, 1).contiguous()

    def encode_clip(
        self,
        video_path: str | Path,
        rotate_90_cw: bool = False,
    ) -> dict[str, torch.Tensor | list[int]]:
        """Encode one clip into the feature structure expected by the StreamVLM adapter."""
        sampled_frames = list(
            self._iter_sampled_frames(
                video_path=video_path,
                rotate_90_cw=rotate_90_cw,
            )
        )
        if not sampled_frames:
            raise ValueError(f"No frames sampled for clip {video_path}")

        self.reset_stream_state()
        warmup_frame = sampled_frames[0]
        for _ in range(self.warmup_frames):
            self._consume_frame(warmup_frame, capture_output=False)

        feature_steps: list[torch.Tensor] = []
        for frame in sampled_frames:
            step_features = self._consume_frame(frame, capture_output=True)
            if step_features is not None:
                feature_steps.append(step_features.cpu())

        if not feature_steps:
            while not feature_steps:
                step_features = self._consume_frame(sampled_frames[-1], capture_output=True)
                if step_features is not None:
                    feature_steps.append(step_features.cpu())

        self.reset_stream_state()
        feature_maps = torch.cat(feature_steps, dim=0)
        return {
            "feats": self._format_feature_maps(feature_maps),
            "spatial_res": list(self.spatial_res),
        }
