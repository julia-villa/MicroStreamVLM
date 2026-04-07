# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Video-frame Pre-processing Utils."""

from typing import Optional

import cv2
import numpy as np


class ConvertBGR2RGB:
    """
    Convert BGR video to RGB video.
    """

    def __call__(self, img: np.array) -> np.array:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Resize:
    """Resize the input image to the given size.

    :param height:
        Height of output image
    :param width:
        Width of output image
    :param interpolation:
        Desired interpolation method. Default is ``cv2.INTER_LINEAR``
    :param keep_aspect_ratio:
           Whether to preserve original aspect ratio when resizing
    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: Optional[object] = cv2.INTER_LINEAR,
        keep_aspect_ratio: Optional[bool] = True,
    ) -> None:
        self.size = (width, height)
        self.interpolation = interpolation
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, img: np.array) -> np.array:
        """
        :param img:
            Image to be scaled.

        :return:
            Rescaled image.
        """
        size = self.get_new_size(img.shape[0], img.shape[1])
        return cv2.resize(img, size, interpolation=self.interpolation)

    def get_new_size(self, height: int, width: int) -> tuple[int, int]:
        """
        :param height:
            Original height of the input image.
        :param width:
            Original width of the input image.

        :return:
            Size of the rescaled image.
        """
        if not self.keep_aspect_ratio:
            return self.size
        ratio = min([self.size[0] / width, self.size[1] / height])
        return int(ratio * width), int(ratio * height)


class Permute:
    """Permute the dimensions of the input.

    :param axes:
        Permute the axes according to the values given.
    """

    def __init__(self, axes: list[int]) -> None:
        self.axes = axes

    def __call__(self, img: np.array) -> np.array:
        """
        :param img:
            The input image.

        :return:
            Image with permuted axes.
        """
        return np.transpose(img, self.axes)


class RescalePixelValues:
    """Divides pixel values by the given scaling factor. Will also convert the input tensor to
    float32 precision.

    :param scale:
        Desired scaling factor. Default is 1.
    """

    def __init__(self, scale: Optional[float] = 1.0) -> None:
        self.scale = scale

    def __call__(self, img: np.array) -> np.array:
        """
        :param img:
            The input image.

        :return:
            Image with rescaled pixel values.
        """
        return np.float32(img) / self.scale


class Reshape:
    """Gives a new shape to an array without changing its data.

    :param shape:
        Desired new shape.
    """

    def __init__(self, shape: list[int]) -> None:
        self.shape = shape

    def __call__(self, img: np.array) -> np.array:
        """
        :param img:
            The input image.

        :return:
            Reshaped image.
        """
        return np.reshape(img, self.shape)


class CropToRectangle:
    """Crop in the middle of the frame to fit the specified aspect ratio (height / width)

    :param aspect_ratio:
        Aspect ratio of the output video.
    """

    def __init__(self, aspect_ratio: Optional[float] = 1.4):
        self.aspect_ratio = aspect_ratio

    def __call__(self, img: np.array) -> np.array:
        """
        :param img:
            The input image.

        :return:
            Cropped image.
        """
        tly, bry, tlx, brx = self.get_cut_coordinates(img)
        return img[tly:bry, tlx:brx]

    def get_cut_coordinates(self, img: np.array) -> tuple[int, int, int, int]:
        """
        :param img:
            The input image.

        :return:
            Coordinates to crop the input image.
        """
        height, width, _ = img.shape
        current_aspect_ratio = height / width
        if current_aspect_ratio > self.aspect_ratio:
            # video is too narrow
            new_height = int(width * self.aspect_ratio)
            offset = int((height - new_height) / 2)
            return offset, offset + new_height, 0, width

        if current_aspect_ratio < self.aspect_ratio:
            # video is too wide
            new_width = int(height / self.aspect_ratio)
            offset = int((width - new_width) / 2)
            return (
                0,
                height,
                offset,
                offset + new_width,
            )

        return 0, height, 0, width
