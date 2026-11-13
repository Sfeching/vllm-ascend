# SPDX-License-Identifier: Apache-2.0
"""Batch-scoped MoonViT RoPE tables and opt-in attention wrappers."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import torch


@dataclass(frozen=True)
class KimiViTRopeTables:
    cos: torch.Tensor
    sin: torch.Tensor
    mode: str
    tokens: int
    head_dim: int

    def __post_init__(self):
        if self.mode not in ("cache", "fused"):
            raise ValueError(f"Unsupported Kimi ViT RoPE mode: {self.mode!r}")
        if self.tokens < 0 or self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("RoPE tables require nonnegative tokens and a positive even head dimension")
        shape = (1, self.tokens, 1, self.head_dim) if self.mode == "cache" else (self.tokens, self.head_dim // 2)
        for table in (self.cos, self.sin):
            if table.shape != shape or table.dtype != torch.float32 or not table.is_contiguous():
                raise ValueError("RoPE tables must be contiguous FP32 tensors with the mode-specific shape")
        if self.cos.device != self.sin.device:
            raise ValueError("Cosine and sine tables must be on the same device")


def prepare_kimi_vit_rope(freqs_cis: torch.Tensor, mode: str) -> KimiViTRopeTables:
    if mode not in ("cache", "fused"):
        raise ValueError(f"Unsupported Kimi ViT RoPE mode: {mode!r}")
    if freqs_cis.ndim != 2 or not freqs_cis.is_complex():
        raise ValueError("Kimi ViT RoPE expects complex position frequencies with shape [tokens, head_dim/2]")
    tokens, half_dim = freqs_cis.shape
    if half_dim == 0:
        raise ValueError("Kimi ViT RoPE requires a nonzero head dimension")
    # Keep FP32 even when the projection uses BF16/FP16. These tensors live
    # only for this encoder call; equal-sized but different grids never alias.
    cos = freqs_cis.real.float().contiguous()
    sin = freqs_cis.imag.float().contiguous()
    head_dim = half_dim * 2
    if mode == "cache":
        cos = cos.repeat_interleave(2, dim=-1).reshape(1, tokens, 1, head_dim)
        sin = sin.repeat_interleave(2, dim=-1).reshape(1, tokens, 1, head_dim)
    return KimiViTRopeTables(cos, sin, mode, tokens, head_dim)


def apply_kimi_vit_rope(
    qkv: torch.Tensor, tables: KimiViTRopeTables
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if qkv.ndim != 4 or qkv.shape[1] != 3 or qkv.shape[0] != tables.tokens or qkv.shape[-1] != tables.head_dim:
        raise ValueError("Packed QKV shape must be [tokens, 3, heads, head_dim] matching the RoPE tables")
    if qkv.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Kimi ViT RoPE supports FP32, FP16 and BF16 projections")
    if qkv.shape[2] <= 0:
        raise ValueError("Packed QKV must contain at least one attention head (heads > 0)")
    if qkv.device != tables.cos.device:
        raise ValueError("QKV and RoPE tables must be on the same device")
    q, k, v = qkv.unbind(dim=1)
    if qkv.numel() == 0:
        return q.contiguous(), k.contiguous(), v
    if tables.mode == "fused":
        # Import the NPU kernel only in workers that explicitly enable it.
        from vllm_ascend.ops.triton.kimi_vit_rope import fused_kimi_vit_rope

        q, k = fused_kimi_vit_rope(qkv, tables.cos, tables.sin)
    else:
        import torch_npu

        q = (
            torch_npu.npu_rotary_mul(q.float().unsqueeze(0), tables.cos, tables.sin, rotary_mode="interleave")
            .squeeze(0)
            .to(qkv.dtype)
        )
        k = (
            torch_npu.npu_rotary_mul(k.float().unsqueeze(0), tables.cos, tables.sin, rotary_mode="interleave")
            .squeeze(0)
            .to(qkv.dtype)
        )
    return q, k, v


def wrap_kimi_vit_encoder_forward(forward: Callable, get_mode: Callable[[], str]) -> Callable:
    if getattr(forward, "_ascend_kimi_vit_rope_wrapped", False) is True:
        return forward

    @wraps(forward)
    def wrapped(self, hidden_states, grid_thws, *, encoder_metadata=None):
        mode = get_mode()
        if mode == "off":
            return forward(self, hidden_states, grid_thws, encoder_metadata=encoder_metadata)
        if mode not in ("cache", "fused"):
            raise ValueError(f"Unsupported Kimi ViT RoPE mode: {mode!r}")
        if encoder_metadata is None:
            if grid_thws is None:
                raise ValueError("Kimi ViT requires a grid or precomputed encoder metadata")
            grid = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid, device=hidden_states.device)
        # Only the ephemeral mapping passed through the layer stack contains
        # this object. Caller-owned metadata and graph input tensors stay intact.
        metadata = dict(encoder_metadata)
        metadata["rope_freqs_cis"] = prepare_kimi_vit_rope(metadata["rope_freqs_cis"], mode)
        return forward(self, hidden_states, grid_thws, encoder_metadata=metadata)

    wrapped._ascend_kimi_vit_rope_wrapped = True
    return wrapped


def wrap_kimi_vit_attention(forward: Callable) -> Callable:
    if getattr(forward, "_ascend_kimi_vit_rope_wrapped", False) is True:
        return forward

    @wraps(forward)
    def wrapped(self, x, cu_seqlens, rope_freqs_cis, max_seqlen=None, sequence_lengths=None):
        if not isinstance(rope_freqs_cis, KimiViTRopeTables):
            return forward(self, x, cu_seqlens, rope_freqs_cis, max_seqlen, sequence_lengths)
        qkv, _ = self.wqkv(x)
        qkv = qkv.view(x.shape[0], 3, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        q, k, v = apply_kimi_vit_rope(qkv, rope_freqs_cis)
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        output = self.attn(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output = output.reshape(
            x.shape[0], self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
        )
        output, _ = self.wo(output)
        return output

    wrapped._ascend_kimi_vit_rope_wrapped = True
    return wrapped
