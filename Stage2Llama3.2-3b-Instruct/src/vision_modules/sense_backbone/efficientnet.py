"""
MIT License

Copyright (c) 2020 Twenty Billion Neurons GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch.nn as nn

from .mobilenet import ConvReLU, InvertedResidual, StridedInflatedMobileNetV2


class StridedInflatedEfficientNet(StridedInflatedMobileNetV2):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(
            ConvReLU(3, 32, 3, stride=2),
            InvertedResidual(32, 24, 3, spatial_stride=1),
            InvertedResidual(24, 32, 3, spatial_stride=2, expand_ratio=6),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(32, 32, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(32, 56, 5, spatial_stride=2, expand_ratio=6),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(56, 56, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(56, 112, 3, spatial_stride=2, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6, temporal_shift=True, temporal_stride=True),
            InvertedResidual(112, 112, 3, spatial_stride=1, expand_ratio=6),
            InvertedResidual(112, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(160, 160, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(160, 272, 5, spatial_stride=2, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6, temporal_shift=True),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 272, 5, spatial_stride=1, expand_ratio=6),
            InvertedResidual(272, 448, 3, spatial_stride=1, expand_ratio=6),
            ConvReLU(448, 1280, 1)
        )
