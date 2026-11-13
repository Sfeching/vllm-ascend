# SPDX-License-Identifier: Apache-2.0
"""Experimental full-row Q/K rotation with contiguous V copy in one launch."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_TOKEN_TILE = 8
_FP32_TOKEN_TILE = 4
_HEAD_TILE = 4
_MAX_TILE_ELEMENTS = 4096


@triton.jit
def _kimi_vit_qkv_kernel(
    QKV,
    COS,
    SIN,
    OUT,
    N_TOKENS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_QKV: tl.constexpr,
    STRIDE_HEAD: tl.constexpr,
    STRIDE_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK_TOKENS
    head_offset = tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, BLOCK_DIM)
    half = tl.arange(0, BLOCK_DIM // 2)
    swap_index = tl.broadcast_to((dim ^ 1)[None, None, :], (BLOCK_TOKENS, BLOCK_HEADS, BLOCK_DIM))
    while start < N_TOKENS:
        token = start + tl.arange(0, BLOCK_TOKENS)
        cos = tl.load(
            COS + token[:, None] * (DIM // 2) + half[None, :],
            (token[:, None] < N_TOKENS) & (half[None, :] < DIM // 2),
            other=0,
        )
        sin = tl.load(
            SIN + token[:, None] * (DIM // 2) + half[None, :],
            (token[:, None] < N_TOKENS) & (half[None, :] < DIM // 2),
            other=0,
        )
        cos_row = tl.join(cos, cos).reshape(BLOCK_TOKENS, BLOCK_DIM)
        signed_sin_row = tl.join(sin, -sin).reshape(BLOCK_TOKENS, BLOCK_DIM)
        for head_group in range(tl.cdiv(HEADS, BLOCK_HEADS)):
            head = head_group * BLOCK_HEADS + head_offset
            mask = (token[:, None, None] < N_TOKENS) & (dim[None, None, :] < DIM)
            if HEADS % BLOCK_HEADS:
                mask = mask & (head[None, :, None] < HEADS)
            for q_or_k in range(2):
                src = (
                    token[:, None, None] * STRIDE_TOKEN
                    + q_or_k * STRIDE_QKV
                    + head[None, :, None] * STRIDE_HEAD
                    + dim[None, None, :] * STRIDE_DIM
                )
                value = tl.load(QKV + src, mask, other=0)
                swapped = tl.gather(value, swap_index, 2)
                rotated = value.to(tl.float32) * cos_row[:, None, :] - (
                    swapped.to(tl.float32) * signed_sin_row[:, None, :]
                )
                dst = (
                    (q_or_k * N_TOKENS + token[:, None, None]) * (HEADS * DIM)
                    + head[None, :, None] * DIM
                    + dim[None, None, :]
                )
                tl.store(OUT + dst, rotated.to(OUT.dtype.element_ty, fp_downcast_rounding="rtne"), mask)
            # Reuse the same token/head layout; V has no rotation or cast.
            src_v = (
                token[:, None, None] * STRIDE_TOKEN
                + 2 * STRIDE_QKV
                + head[None, :, None] * STRIDE_HEAD
                + dim[None, None, :] * STRIDE_DIM
            )
            dst_v = (
                (2 * N_TOKENS + token[:, None, None]) * (HEADS * DIM) + head[None, :, None] * DIM + dim[None, None, :]
            )
            tl.store(OUT + dst_v, tl.load(QKV + src_v, mask, other=0), mask)
        start += tl.num_programs(0) * BLOCK_TOKENS


def fused_kimi_vit_qkv(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    tokens, _, heads, dim = qkv.shape
    output = torch.empty((3, tokens, heads, dim), dtype=qkv.dtype, device=qkv.device)
    if tokens:
        init_device_properties_triton()
        block_dim = triton.next_power_of_2(dim)
        tile = _FP32_TOKEN_TILE if qkv.dtype == torch.float32 else _TOKEN_TILE
        max_elements = _MAX_TILE_ELEMENTS // (2 if qkv.dtype == torch.float32 else 1)
        tile = min(tile, max(1, max_elements // (_HEAD_TILE * block_dim)))
        programs = min(triton.cdiv(tokens, tile), get_vectorcore_num())
        _kimi_vit_qkv_kernel[(programs,)](
            qkv,
            cos,
            sin,
            output,
            tokens,
            heads,
            dim,
            *qkv.stride(),
            BLOCK_TOKENS=tile,
            BLOCK_HEADS=_HEAD_TILE,
            BLOCK_DIM=block_dim,
            enable_fp_fusion=False,
        )
    return output[0], output[1], output[2]
