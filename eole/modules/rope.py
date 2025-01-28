import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple


# Help functions for Rotary Embeddings
def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    query: Tensor, key: Tensor, rope: Tuple[Tensor, Tensor], interleave: bool
) -> Tuple[Tensor, Tensor]:
    # now rope is a tuple (cos, sin)
    cos, sin = rope
    if interleave:
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)

        # Reshape cos and sin to match the dimensions of query_ and key_
        cos = cos[:, : cos.size(1) // 2].view(1, query_.size(1), 1, query_.size(3))
        sin = sin[:, : sin.size(1) // 2].view(1, key_.size(1), 1, key_.size(3))

        query_rotated = query_[..., 0] * cos - query_[..., 1] * sin
        query_rotated_imag = query_[..., 0] * sin + query_[..., 1] * cos
        query_out = torch.stack((query_rotated, query_rotated_imag), dim=-1).flatten(3)

        key_rotated = key_[..., 0] * cos - key_[..., 1] * sin
        key_rotated_imag = key_[..., 0] * sin + key_[..., 1] * cos
        key_out = torch.stack((key_rotated, key_rotated_imag), dim=-1).flatten(3)

        return query_out.transpose(1, 2).type_as(query), key_out.transpose(1, 2).type_as(key)

        # Old code with complex instead
        # rope_complex = torch.complex(cos, sin)
        # query_ = torch.view_as_complex(query_)
        # key_ = torch.view_as_complex(key_)
        # query_out = torch.view_as_real(query_ * rope_complex).flatten(3)
        # key_out = torch.view_as_real(key_ * rope_complex).flatten(3)
        # return query_out.transpose(1, 2).type_as(query), key_out.transpose(
        #     1, 2
        # ).type_as(key)
    else:
        rotary_dim = cos.size(1)
        head_dim = query.size(3)
        if rotary_dim < head_dim:
            q_embed = (query[:, :, :, :rotary_dim] * cos) + (rotate_half(query[:, :, :, :rotary_dim]) * sin)
            k_embed = (key[:, :, :, :rotary_dim] * cos) + (rotate_half(key[:, :, :, :rotary_dim]) * sin)
            q_embed = torch.cat([q_embed, query[:, :, :, rotary_dim:]], dim=-1)
            k_embed = torch.cat([k_embed, key[:, :, :, rotary_dim:]], dim=-1)
        else:
            q_embed = (query * cos) + (rotate_half(query) * sin)
            k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed.type_as(query), k_embed.type_as(key)


class RotaryPosition(nn.Module):
    """
    Handles rotary position embeddings for transformer models.

    This module was refactored from multi-headed attention for improved clarity
    and to support future enhancements, such as additional scaling types.
    """

    def __init__(self, model_config):
        """
        Initializes the RotaryPosition module.

        Args:
            model_config: Configuration object that contains model parameters,
                          including rotary embedding settings.

        Attributes:
            model_config: The configuration object passed during initialization.
            dim_per_head: The dimensionality of each attention head, computed
                          as `hidden_size // heads`.
            rotary_interleave: Boolean flag to determine if head dimensions should
                               be interleaved or split when applying rotary embeddings.
            rotary_theta: The base frequency for rotary embeddings.
            inv_freq: Inverse frequency values used to calculate the rotary embeddings.

        Notes:
            - If `rotary_dim` is set to 0 in the configuration, it defaults to
              `dim_per_head`.
            - Additional scaling types can be added in the future by extending this class.
        """
        super(RotaryPosition, self).__init__()
        self.model_config = model_config
        self.dim_per_head = model_config.dim_per_head
        if model_config.rope_config.rotary_dim == 0:
            rotary_dim = self.dim_per_head
        else:
            rotary_dim = model_config.rope_config.rotary_dim
        self.rotary_interleave = model_config.rope_config.rotary_interleave
        self.rotary_theta = model_config.rope_config.rotary_theta
        self.inv_freq = 1.0 / (self.rotary_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        # TODO: extend with other scaling types
        if getattr(self.model_config.rope_config, "scaling_type", None) == "llama3":
            self.llama3_scaling()
        self.update(1024)

    def llama3_scaling(self):
        """
        Applies the LLaMA3.1-specific scaling to the inverse frequencies.

        This scaling is based on LLaMA3.1's handling of different frequency components
        within rotary embeddings. The method modifies `self.inv_freq` in place.

        Notes:
            - Original values for `factor`, `low_freq_factor`, `high_freq_factor`,
              and `original_max_position_embeddings` are taken from the configuration.
            - The scaling factors are applied conditionally based on the wavelength
              derived from the inverse frequencies.
        """
        rope_config = self.model_config.rope_config
        factor = rope_config.scaling_factor  # `8` in the original implementation
        low_freq_factor = rope_config.low_freq_factor  # `1` in the original implementation
        high_freq_factor = rope_config.high_freq_factor  # `4` in the original implementation
        old_context_len = rope_config.original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / self.inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        self.inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    def update(self, maxseqlen, step=0, prefetch=1024):
        """
        Computes the rotary position embeddings for a given input.

        Args:
            maxseqlen: max seq length of the input embeddings.
            step: The current step or position within the sequence. Defaults to 0.
            offset: An optional offset to apply to the position indices.
                    This is used for the specific `flash_attn_with_kvcache` path,
                    which requires processes by chunks of 32 tokens. Defaults to 0.

        Returns:
            torch.Tensor: A tensor containing the computed rotary embeddings.

        Notes:
            - The returned tensor contains cosine and sine values representing the
              rotary embeddings, concatenated along the last dimension.
            - The output tensor's dimensions are `[maxseqlen, dim]`, where `dim` is
              twice the size of the original inverse frequency tensor (`inv_freq`).
        """
        offset = 32  # make sure we have at least 32 positions for flash_attn_with_kvcache
        if step == 0:
            maxseqlen = 1024  # reset as in init() with self.update(1024)
        elif hasattr(self, "cos") and self.cos.size(0) >= max(offset + (step or 0), 0) + maxseqlen:
            return self.cos, self.sin

        maxseqlen += prefetch
        device = self.cos.device if hasattr(self, "cos") else torch.device("cpu")

        tmax = torch.arange(max(offset + step, 0) + maxseqlen, device=device)
        rope = torch.outer(tmax, self.inv_freq.to(device))
        cos = torch.cos(rope)
        sin = torch.sin(rope)
        cos = torch.cat((cos, cos), dim=-1)  # Double the size by repeating `cos`
        sin = torch.cat((sin, sin), dim=-1)  # Double the size by repeating `sin`

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        return cos, sin
