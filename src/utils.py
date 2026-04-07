# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""VLM Utils."""

from typing import AnyStr

import numpy as np
import torch.nn as nn
from peft.peft_model import PeftModel, PeftModelForCausalLM
from transformers import PreTrainedModel


def get_vlm_lang_handler(model: PreTrainedModel | PeftModel | PeftModelForCausalLM) -> nn.Module:
    """Get correct language decoder

    :param model:
        VLM's language model backbone

    :return:
        Language decoder from the model
    """
    if isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftModel):
        return model.base_model.model.get_decoder()
    elif hasattr(model.model, "decoder"):
        return model.model.decoder
    else:
        return model.model


def load_video_timestamps(file_path: AnyStr) -> np.array:
    """
    Return array of timestamps per video frame (features)

    :param file_path:
        Path of timestamp file.

    :return:
        List of timestamps in UNIX format
    """
    # Transform timestamps to seconds (from nanoseconds)
    timestamps = np.load(file_path).astype(np.double)
    timestamps = np.array([int(xx / 1e9) + (xx / 1e9) % 1 for xx in timestamps]) + 28800.0
    return timestamps
