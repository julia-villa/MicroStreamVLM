# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Custom Dot Product Attention Layer Implementation."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class CustomDotProdXAttn(nn.Module):
    """Custom Dot Product Cross-Attention Layer"""

    def __init__(
        self,
        hidden_dim: int,
        vision_feat_dim: int,
        attn_dim: Optional[int] = 4096,
        num_xattn_heads: Optional[int] = 1,
    ):
        """
        :param hidden_dim:
            Hidden dimension of the transformer model.
        :param vision_feat_dim:
            Dimensionality of the extracted vision features.
        :param attn_dim:
            The attention dimension to use.
        :param num_xattn_heads:
            Number of cross-attention heads.
        """
        super().__init__()
        self.attn_dim = attn_dim
        self.num_xattn_heads = num_xattn_heads

        self.hidden_attn_fc = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.feat_attn_keys_fc = nn.Linear(vision_feat_dim, attn_dim, bias=False)
        self.vision_projector = self._feed_forward_layer(vision_feat_dim, attn_dim)
        self.out_fc = nn.Linear(attn_dim, hidden_dim, bias=False)

        self.softmax = torch.nn.functional.softmax

    @staticmethod
    def _feed_forward_layer(
        in_dim: int, out_dim: int, expansion_factor: Optional[int] = 1
    ) -> nn.Module:
        """Simple feed forward layer with layer normalization and GELU activation."""
        inner_dim = int(in_dim * expansion_factor)
        return nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-3, elementwise_affine=False),
            nn.Linear(in_dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, out_dim, bias=False),
        )

    def _prepare_for_multiheaded_xattn(self, embedding: torch.tensor) -> torch.tensor:
        """Reshape QKV for multi-headed attention."""
        embedding = embedding.reshape(
            embedding.shape[0],
            embedding.shape[1],
            embedding.shape[2],
            self.num_xattn_heads,
            embedding.shape[3] // self.num_xattn_heads,
        )
        embedding = embedding.permute(0, 1, 3, 2, 4)
        return embedding

    def forward(
        self,
        hidden_states: torch.tensor,
        vision_feats: torch.tensor,
        vision_xattn_mask: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Compute custom dot product attention using projected vectors from language and vision
        modalities.

        :param hidden_states:
            LM-backbone hidden states.
        :param vision_feats:
            Vision features.
        :param vision_xattn_mask:
            Mask indicating which tokens correspond to vision tokens.

        :return:
            Tuple of resultant tensor and attention scores.
        """
        assert vision_xattn_mask is not None

        _batch_size = hidden_states.shape[0]
        _hidden_dim = hidden_states.shape[-1]

        # Mask hidden states with vision xattn mask
        hidden_states = hidden_states[
            torch.where(vision_xattn_mask)[0], torch.where(vision_xattn_mask)[1]
        ]
        hidden_states = hidden_states.reshape(_batch_size, -1, _hidden_dim)

        # Reduce vision xattn mask to match reduced hidden state shape
        vision_xattn_mask = vision_xattn_mask[
            torch.where(vision_xattn_mask)[0], torch.where(vision_xattn_mask)[1]
        ]
        vision_xattn_mask = vision_xattn_mask.reshape(_batch_size, -1)

        # Apply linear projection on both hidden states and vision feats
        vision_feats = vision_feats.to(hidden_states)
        query = self.hidden_attn_fc(hidden_states).unsqueeze(2)
        keys = self.feat_attn_keys_fc(vision_feats)
        values = self.vision_projector(vision_feats)

        # Reshape embedded tensors for multi-head attention
        query = self._prepare_for_multiheaded_xattn(query)
        keys = self._prepare_for_multiheaded_xattn(keys)
        values = self._prepare_for_multiheaded_xattn(values)

        # Perform top-down cross-attention from LM hidden states to vision feats
        attn_logits = query @ keys.transpose(3, 4)
        attn_logits = attn_logits / math.sqrt(self.attn_dim)
        attn_logits = attn_logits - attn_logits.amax(dim=-1, keepdim=True).detach()
        attention = self.softmax(attn_logits, dim=-1)

        # Apply attention scores to vision feature value embeddings
        values = attention @ values
        values = values.squeeze(3)
        values = values.reshape(values.shape[0], values.shape[1], -1)
        values = self.out_fc(values).to(hidden_states)
        values = values * (vision_xattn_mask >= 2).unsqueeze(-1)
        return values, attention


class CustomDotProdXAttnModule(nn.Module):
    """Cross-attention wrapper that manages all cross-attention layers."""

    def __init__(
        self,
        hidden_dim: int,
        vision_feat_dim: int,
        xattn_type: Optional[str] = "dotprod",
        attn_dim: Optional[int] = 4096,
        num_xattn_layers: Optional[int] = 1,
        num_xattn_heads: Optional[int] = 32,
    ):
        """
        :param hidden_dim:
            Hidden dimension of the transformer model.
        :param vision_feat_dim:
            Dimensionality of the encoded vision features.
        :param xattn_type:
            Type of cross-attention layer.
        :param attn_dim:
            The attention dimension to use.
        :param num_xattn_layers:
            Number of cross-attention layers to initialize.
        :param num_xattn_heads:
            Number of cross-attention heads per layer.
        """
        super().__init__()
        self.attn_layers = []
        self.num_layers = num_xattn_layers
        assert xattn_type in ["dotprod"]

        d = dict(
            hidden_dim=hidden_dim,
            vision_feat_dim=vision_feat_dim,
            attn_dim=attn_dim,
            num_xattn_heads=num_xattn_heads,
        )

        for _ in range(self.num_layers):
            self.attn_layers.append(CustomDotProdXAttn(**d))

        self.attn_layers = nn.ModuleList(self.attn_layers)

    def forward(
        self,
        hidden_states: torch.tensor,
        vision_feats: torch.tensor,
        vision_xattn_mask: torch.tensor,
        **kwargs
    ) -> torch.tensor:
        """Forward pass through each cross-attention layers and return the values from the final
        layer.
        """
        values = None

        # Process stack layer-wise
        for layer_id in range(self.num_layers):
            values, _ = self.attn_layers[layer_id](
                hidden_states, vision_feats, vision_xattn_mask, **kwargs
            )
            hidden_states = values

        return values
