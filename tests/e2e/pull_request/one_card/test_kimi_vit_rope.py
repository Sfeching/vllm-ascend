# SPDX-License-Identifier: Apache-2.0
"""A2/A3 acceptance: compare the new paths with the actual baseline NPU op."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.kimi_k25_vit import MoonViT3dEncoder, MoonViTEncoderLayer, Rope2DPosEmbRepeated

from vllm_ascend.ops.kimi_vit_rope import (
    apply_kimi_vit_rope,
    prepare_kimi_vit_rope,
)
from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")


def make_baseline_op():
    with set_current_vllm_config(VllmConfig()):
        return AscendApplyRotaryEmb(enforce_enable=True, is_neox_style=False, enable_fp32_compute=True)


@pytest.mark.parametrize("mode", ["cache", "fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 17, 21904, 43808])
@torch.inference_mode()
def test_kimi_vit_rope_matches_baseline_npu(mode, dtype, tokens):
    torch.manual_seed(19)
    packed = torch.randn(tokens, 3, 12, 128, dtype=dtype, device="npu")
    original = packed.clone()
    # Generate positions on CPU: complex tensor support is not required here.
    freqs = torch.polar(torch.ones(tokens, 64), torch.randn(tokens, 64))
    tables_cpu = prepare_kimi_vit_rope(freqs, mode)
    tables = type(tables_cpu)(tables_cpu.cos.npu(), tables_cpu.sin.npu(), mode, tokens, 128)
    cos, sin = freqs.real.contiguous().npu(), freqs.imag.contiguous().npu()
    baseline = make_baseline_op()
    if tokens == 0:
        q, k, v = apply_kimi_vit_rope(packed, tables)
        assert q.shape == k.shape == v.shape == (0, 12, 128)
        return
    expected_q = baseline.forward_oot(packed[:, 0], cos, sin)
    expected_k = baseline.forward_oot(packed[:, 1], cos, sin)
    q, k, v = apply_kimi_vit_rope(packed, tables)
    torch.npu.synchronize()
    # Both paths preserve separate FP32 rounding, including cancellation cases.
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert v.data_ptr() == packed[:, 2].data_ptr()
    assert q.is_contiguous() and k.is_contiguous()


@pytest.mark.parametrize("mode", ["cache", "fused"])
@torch.inference_mode()
def test_kimi_vit_rope_strided_bf16_projection(mode):
    torch.manual_seed(31)
    packed = torch.randn(17, 3, 12, 256, dtype=torch.bfloat16, device="npu")[..., ::2]
    freqs = torch.polar(torch.ones(17, 64), torch.randn(17, 64))
    tables_cpu = prepare_kimi_vit_rope(freqs, mode)
    tables = type(tables_cpu)(tables_cpu.cos.npu(), tables_cpu.sin.npu(), mode, 17, 128)
    baseline = make_baseline_op()
    q, k, v = apply_kimi_vit_rope(packed, tables)
    for actual, index in ((q, 0), (k, 1)):
        expected = baseline.forward_oot(packed[:, index], freqs.real.contiguous().npu(), freqs.imag.contiguous().npu())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert v.data_ptr() == packed[:, 2].data_ptr()


@pytest.mark.parametrize(
    "tokens,heads,dim,dtype,strided",
    [
        (17, 5, 128, torch.float32, False),
        (17, 5, 128, torch.bfloat16, True),
        (5, 3, 8, torch.float16, False),
        (2051, 1, 4, torch.float32, False),
    ],
)
@torch.inference_mode()
def test_fused_partial_head_group_and_token_tail(tokens, heads, dim, dtype, strided):
    """Head and token tails must not read another Q/K/V row or overwrite input."""
    torch.manual_seed(41)
    packed = torch.randn(tokens, 3, heads, dim * (2 if strided else 1), dtype=dtype, device="npu")
    if strided:
        packed = packed[..., ::2]
    original = packed.clone()
    freqs = torch.polar(torch.ones(tokens, dim // 2), torch.randn(tokens, dim // 2))
    tables_cpu = prepare_kimi_vit_rope(freqs, "fused")
    tables = type(tables_cpu)(tables_cpu.cos.npu(), tables_cpu.sin.npu(), "fused", tokens, dim)
    baseline = make_baseline_op()
    q, k, v = apply_kimi_vit_rope(packed, tables)
    for actual, index in ((q, 0), (k, 1)):
        expected = baseline.forward_oot(packed[:, index], freqs.real.contiguous().npu(), freqs.imag.contiguous().npu())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous()
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert v.data_ptr() == packed[:, 2].data_ptr()


@pytest.mark.parametrize("mode", ["cache", "fused"])
@pytest.mark.parametrize("grids", [[[1, 2, 3]], [[1, 3, 2]], [[1, 2, 3], [1, 3, 2]], [[2, 3, 4]]])
@torch.inference_mode()
def test_kimi_attention_wrapper_matches_upstream_attention(mode, grids):
    """Use the installed worker patch, real H/W/T frequencies and actual FIA."""
    import vllm_ascend.patch.worker.patch_kimi_k25  # noqa: F401

    torch.manual_seed(23)
    lengths = [h * w for t, h, w in grids for _ in range(t)]
    tokens, heads, dim = sum(lengths), 12, 128
    packed = torch.randn(tokens, 3, heads, dim, dtype=torch.bfloat16, device="npu")
    position = Rope2DPosEmbRepeated(dim, max_height=4, max_width=4)
    freqs = position.get_freqs_cis(grids, torch.device("cpu")).npu()
    baseline = make_baseline_op()
    with set_current_vllm_config(VllmConfig()):
        attention = AscendMMEncoderAttention(num_heads=heads, head_size=dim)

    layer = SimpleNamespace(
        wqkv=lambda x: (packed.flatten(1), None),
        wo=lambda x: (x, None),
        num_attention_heads_per_partition=heads,
        hidden_size_per_attention_head=dim,
        apply_rotary_emb=baseline.forward_oot,
        attn=attention.forward_oot,
    )
    x = torch.empty(tokens, 1024, device="npu")
    cumulative = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to("npu", dtype=torch.int32)
    maximum = torch.tensor(max(lengths))
    installed = MoonViTEncoderLayer.attention_qkvpacked
    assert installed._ascend_kimi_vit_rope_wrapped is True
    assert MoonViT3dEncoder.forward._ascend_kimi_vit_rope_wrapped is True
    upstream = installed.__wrapped__
    expected = upstream(layer, x, cumulative, freqs, maximum)
    tables = prepare_kimi_vit_rope(freqs, mode)
    actual = installed(layer, x, cumulative, tables, maximum)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
