# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""StridedInflatedEfficientNet-based Vision Model and Utils."""

from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose

from src.vision_modules.sense_backbone import StridedInflatedEfficientNet
from src.vision_modules.utils import (
    ConvertBGR2RGB,
    CropToRectangle,
    Permute,
    RescalePixelValues,
    Reshape,
    Resize,
)


class Hypermodel:
    """
    3D-CNN-based vision backbone based on the StridedInflatedEfficientNet architecture from:
    https://github.com/quic/sense

    :param num_global_classes:
        Number of classes output classes
    :param num_frames_required:
        Minimum number of frames required to step the model
    :param path_weights:
        Path to weights file
    :param gpus:
        List of GPUs to use
    :param half_precision:
        Whether to use half precision
    """

    def __init__(
        self,
        num_global_classes: int,
        num_frames_required: Optional[int] = 4,
        path_weights: Optional[str] = None,
        gpus: Optional[list[int]] = None,
        half_precision: Optional[bool] = False,
    ):
        super().__init__()

        self.num_global_classes = num_global_classes
        self.num_frames_required = num_frames_required

        self.path_weights = path_weights
        self.gpus = gpus
        self.half_precision = half_precision

        # will be created when initialize is called
        self.features = None
        self.classifier = None
        self.net = None
        self.transforms = None
        self._buffer_frames = []

    def __call__(self, frame: np.array) -> Union[torch.tensor, None]:
        """
        Preprocesses frames, stores them into a buffer and performs inference on the whole
        buffer when it is full (i.e. when its length is equal to self.num_frames_required)

        :param frame:
            Single frame. Typically, a uint8 bit numpy array of size H x W x 3.

        :return:
            Set of predictions when the frame buffer is full, None otherwise
        """
        self._buffer_frames.append(self.transforms(frame))

        # Process if buffer is full
        if len(self._buffer_frames) == self.num_frames_required:
            # Stack frames and convert to torch
            frames = np.concatenate(self._buffer_frames, axis=0)
            # Shape should now be batch x channels x time x height x width
            frames = torch.from_numpy(frames)
            if self.half_precision:
                frames = frames.half()

            # Inference
            with torch.no_grad():
                prediction = self.forward(frames)

            # Empty buffer
            self._buffer_frames = []

            # Convert and return
            return prediction

        return None

    def load_weights(self, path_weights: str) -> None:
        """Method to load model weights."""
        load_hypermodel_weights(self.net, path_weights, strict=False)

    def initialize(self):  # noqa: D102
        """Initialize the 3D CNN, data preprocessing pineline and load model weights."""
        # create a prediction stream
        self.gpus = self.gpus if torch.cuda.is_available() else None
        self.features = StridedInflatedEfficientNet()
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, self.num_global_classes))
        self.net = nn.Sequential(self.features, MeanModule(), self.classifier)

        # preprocessing
        self.transforms = [
            ConvertBGR2RGB(),
            CropToRectangle(aspect_ratio=1.4),
            Resize(height=224, width=160, keep_aspect_ratio=False),
            RescalePixelValues(scale=255.0),
            Permute([2, 0, 1]),
            Reshape([1, 3, 224, 160]),
        ]
        if self.gpus is not None:
            self.net = nn.DataParallel(self.net, device_ids=self.gpus).cuda(self.gpus[0])
        else:
            self.net = self.net

        # preprocessing functions
        self.transforms = Compose(self.transforms)
        # load weights
        if self.path_weights is not None:
            self.load_weights(self.path_weights)
        # set to eval mode
        self.half_precision = self.half_precision
        if self.half_precision:
            self.net.half()
        self.net.eval()

    def forward(self, frames):  # noqa: D102
        """Perform a forward pass of the 3D CNN."""
        # Expected input size: Frames x Channels x Height x Width
        return self.net(frames).cpu().numpy()

    def reset_buffer(self):
        """Reset the internal frames buffer to process a new video"""
        self._buffer_frames = []


class MeanModule(nn.Module):
    """nn Module to compute mean over last two dimensions."""

    def __init__(self):
        super().__init__()

    def forward(self, xx: torch.Tensor) -> torch.Tensor:
        """Return the mean over the last two dimensions."""
        return xx.mean(dim=-1).mean(dim=-1)


class HypermodelFeatureProcessor(nn.Module):
    """Prepare the features from backbone model"""

    def __init__(self):
        super().__init__()
        self.encoder_feat_dim = 1280

    def get_embed_dim(self):
        """Returns the dimensionality of the model features."""
        return self.encoder_feat_dim

    def forward(self, video: torch.Tensor) -> dict[str, Union[torch.Tensor, list[int]]]:
        """Preprocess the video featuers."""
        # video: [B, L, C, H, W]
        video = video.flatten(3, 4).permute(0, 1, 3, 2)  # [B, L, H*W, C]
        video = torch.concatenate(
            [
                torch.zeros(video.shape[0], 1, video.shape[2], video.shape[3]).to(video),
                video,
            ],
            dim=1,
        )
        return {"feats": video, "spatial_res": [5, 7]}


def load_hypermodel_weights(
    net: Hypermodel, path: str, strict: Optional[bool] = True
) -> Hypermodel:
    """Load vision backbone weights from .pth file into model.

    :param net:
        Hypermodel instance
    :param path:
        Path to weights
    :param strict:
        Whether to enforce strict key mapping between the checkpoint and model

    :return:
        Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    if not isinstance(net, nn.DataParallel):
        # Remove 'module.' in all checkpoint keys
        checkpoint = {key.replace("module.", ""): value for (key, value) in checkpoint.items()}

    if not strict:
        # Search for missing keys or dimension mismatch
        to_pop = []
        to_add = []
        for (key, weight), (key_ckpt, weight_ckpt) in zip(
            net.state_dict().items(), checkpoint.items()
        ):
            if key not in checkpoint:
                if weight_ckpt.shape == weight.shape:
                    to_add.append((key, weight_ckpt))
                    to_pop.append(key_ckpt)

            elif checkpoint[key].shape != weight.shape:
                to_pop.pop(key)

        for key, value in to_add:
            checkpoint[key] = value

        for key in to_pop:
            checkpoint.pop(key)

    net.load_state_dict(checkpoint, strict=strict)
    return net
