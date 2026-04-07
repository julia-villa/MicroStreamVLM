# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Vision Backbone to LLM Cross-attention Adapter Implementation."""

from typing import Any, Callable, Optional, Union

import torch
from torch import nn


class XAttnAdapter(nn.Module):
    """
    Cross-attention adapter between the vision and language backbones.

    :param model_lang_embed_tokens_layer:
        The token embedding layer object from language backbone
    :param injection_layer_ids:
        Layer number(s) of the language model where the vision features should be cross attended to
    """

    def __init__(
        self,
        model_lang_embed_tokens_layer: Callable,
        injection_layer_ids: list[int | str],
    ):
        super().__init__()
        # All layer ids must be strings
        injection_layer_ids = list(map(str, injection_layer_ids))

        self.injection_layer_ids = injection_layer_ids
        self.embed_tokens = model_lang_embed_tokens_layer

    def _embed_text_tokens(self, text_tokens: torch.tensor) -> dict[str, torch.tensor]:
        """Return a tensor containing embeddings of tokenized text (input ids)."""
        return {"0": self.embed_tokens(text_tokens)}

    def _combine_multilayer_embedding_dicts(
        self,
        multilayer_video_feats: dict[str, torch.tensor],
        text_tokens: torch.tensor,
        multilayer_embedded_text_tokens: dict[str, torch.tensor],
    ) -> dict[str, dict[str, Any] | dict[str, Any] | Any]:
        """
        Prepare a combined dictionary with both visual and textual inputs.

        :param multilayer_video_feats:
            Dictionary of tensors with different layers' vision feats
        :param text_tokens:
            Tensor of shape [batch_size, sequence_length].
        :param multilayer_embedded_text_tokens:
            Dictionary of tensors with different layers' text embeddings.

        :return:
            Multilayer dictionary with the combined features for all layers.
        """

        final_multilayer_embeddings = {
            "text_tokens": text_tokens,
            "0": {"comb": multilayer_embedded_text_tokens["0"]},
        }

        # Update dict with vision features from each layer
        for layer_id in self.injection_layer_ids:
            final_multilayer_embeddings[layer_id] = {"vision": multilayer_video_feats[layer_id]}

        return final_multilayer_embeddings

    def _init_multilayer_vision_feats_dict(
        self, vision_feats: torch.tensor
    ) -> dict[str, torch.tensor]:
        """Initialize multi-layer vision features dict"""
        multilayer_vision_feats = {}
        for layer_id in self.injection_layer_ids:
            if layer_id == "0":
                multilayer_vision_feats[layer_id] = vision_feats.mean(dim=2)
            else:
                multilayer_vision_feats[layer_id] = vision_feats
        return multilayer_vision_feats

    def forward(
        self,
        vision_feats: torch.tensor,
        text_tokens: torch.tensor,
        vision_xattn_mask: Optional[torch.tensor] = None,
        buffer_xattn_mask: Optional[torch.tensor] = None,
    ) -> dict[str, Union[str, torch.tensor, None]]:
        """
        :param vision_feats:
            Vision features extracted from a vision encoder.
        :param text_tokens:
            Tokenized text
        :param vision_xattn_mask:
            Mask indicating which parts of the vision features are valid
        :param buffer_xattn_mask:
            Buffer mask that indicates which elements are part of the same utterance

        :return:
            embedding dict
        """
        vision_feats = vision_feats["feats"] if isinstance(vision_feats, dict) else vision_feats

        # Prepare embedding dicts
        embedded_vision_feats = self._init_multilayer_vision_feats_dict(
            vision_feats
        )  # output shape: (batch_size, seq_len_padded, emdb_dims)
        embedded_text_tokens = self._embed_text_tokens(
            text_tokens
        )  # output shape: (batch_size, seq_len_padded, emdb_dims)

        # Combine both modalities
        final_embedding = self._combine_multilayer_embedding_dicts(
            embedded_vision_feats, text_tokens, embedded_text_tokens
        )

        final_embedding["type"] = "xattn"
        final_embedding["vision_xattn_mask"] = vision_xattn_mask
        final_embedding["buffer_xattn_mask"] = buffer_xattn_mask

        return final_embedding
