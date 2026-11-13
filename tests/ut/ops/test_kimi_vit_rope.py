# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; only the NPU rotary primitive is replaced."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_vit_rope.py"
spec = importlib.util.spec_from_file_location("kimi_vit_rope_under_test", SOURCE)
rope = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rope
spec.loader.exec_module(rope)


def reference(x, freqs):
    pairs = x.float().contiguous().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.view_as_complex(pairs) * freqs[:, None, :]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


def rotary_primitive(x, cos, sin, *, rotary_mode):
    assert x.dtype == cos.dtype == sin.dtype == torch.float32
    assert rotary_mode == "interleave"
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_cached_rope_matches_complex_reference_without_modifying_qkv(monkeypatch, dtype, tokens):
    torch.manual_seed(7)
    packed = torch.randn(tokens, 3, 12, 128, dtype=dtype)
    original = packed.clone()
    freqs = torch.polar(torch.ones(tokens, 64), torch.randn(tokens, 64))
    tables = rope.prepare_kimi_vit_rope(freqs, "cache")
    op = Mock(side_effect=rotary_primitive)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=op))
    q, k, v = rope.apply_kimi_vit_rope(packed, tables)
    torch.testing.assert_close(q, reference(packed[:, 0], freqs), rtol=0, atol=0)
    torch.testing.assert_close(k, reference(packed[:, 1], freqs), rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert v.data_ptr() == packed[:, 2].data_ptr()
    assert q.is_contiguous() and k.is_contiguous()
    assert op.call_count == (2 if tokens else 0)
    if tokens:
        assert op.call_args_list[0].args[1] is op.call_args_list[1].args[1]


def test_equal_token_counts_do_not_share_position_tables():
    # 2x3 and 3x2 image grids have equal lengths but different coordinates.
    grids = [torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 0, 1, 0, 1])]
    tables = [
        rope.prepare_kimi_vit_rope(torch.polar(torch.ones(6, 4), x[:, None].expand(6, 4).float()), "fused")
        for x in grids
    ]
    assert not torch.equal(tables[0].cos, tables[1].cos)
    assert tables[0].cos.data_ptr() != tables[1].cos.data_ptr()
    assert tables[0].cos.shape == (6, 4)


def test_encoder_prepares_once_and_preserves_caller_metadata():
    freqs = torch.polar(torch.ones(6, 4), torch.randn(6, 4))
    metadata = {"rope_freqs_cis": freqs, "cu_seqlens": torch.tensor([0, 6])}
    seen = []

    def forward(self, hidden, grid, *, encoder_metadata=None):
        for _ in range(27):
            seen.append(encoder_metadata["rope_freqs_cis"])
        return hidden

    encoder = SimpleNamespace(prepare_encoder_metadata=Mock(return_value=metadata))
    wrapped = rope.wrap_kimi_vit_encoder_forward(forward, lambda: "cache")
    hidden = torch.randn(6, 8)
    assert wrapped(encoder, hidden, [[1, 2, 3]]) is hidden
    assert encoder.prepare_encoder_metadata.call_count == 1
    assert all(table is seen[0] for table in seen)
    assert isinstance(seen[0], rope.KimiViTRopeTables)
    assert metadata["rope_freqs_cis"] is freqs
    seen.clear()
    wrapped(encoder, hidden, None, encoder_metadata=metadata)
    assert encoder.prepare_encoder_metadata.call_count == 1
    assert metadata["rope_freqs_cis"] is freqs


def test_disabled_mode_delegates_without_preparing_metadata():
    original = Mock(return_value="original")
    encoder = SimpleNamespace(prepare_encoder_metadata=Mock())
    wrapped = rope.wrap_kimi_vit_encoder_forward(original, lambda: "off")
    assert wrapped(encoder, torch.empty(1), None) == "original"
    encoder.prepare_encoder_metadata.assert_not_called()
    assert original.call_count == 1


@pytest.mark.parametrize("mode", ["unknown", "", "CACHE"])
def test_invalid_mode_fails_before_tensor_work(mode):
    with pytest.raises(ValueError, match="mode"):
        rope.prepare_kimi_vit_rope(torch.ones(2, 4, dtype=torch.complex64), mode)


def test_incompatible_position_shape_is_rejected():
    tables = rope.prepare_kimi_vit_rope(torch.ones(2, 4, dtype=torch.complex64), "cache")
    with pytest.raises(ValueError, match="shape"):
        rope.apply_kimi_vit_rope(torch.ones(3, 3, 2, 8), tables)


@pytest.mark.parametrize("field", ["sin_dtype", "sin_shape", "sin_device", "cos_stride", "mode", "head_dim"])
def test_malformed_tables_fail_before_kernel_launch(field):
    tables = rope.prepare_kimi_vit_rope(torch.ones(2, 4, dtype=torch.complex64), "fused")
    changes = {
        "sin_dtype": {"sin": tables.sin.half()},
        "sin_shape": {"sin": tables.sin[:, :2]},
        "sin_device": {"sin": torch.empty(2, 4, device="meta")},
        "cos_stride": {"cos": torch.ones(2, 8)[:, ::2]},
        "mode": {"mode": "unknown"},
        "head_dim": {"head_dim": 7},
    }
    with pytest.raises(ValueError):
        replace(tables, **changes[field])


def test_zero_heads_is_not_a_valid_empty_batch():
    tables = rope.prepare_kimi_vit_rope(torch.ones(2, 4, dtype=torch.complex64), "fused")
    with pytest.raises(ValueError, match="heads"):
        rope.apply_kimi_vit_rope(torch.empty(2, 3, 0, 8), tables)


def load_kernel(monkeypatch, cores=4):
    import triton
    import triton.language as tl

    monkeypatch.setitem(sys.modules, "vllm.triton_utils", SimpleNamespace(triton=triton, tl=tl))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.triton.triton_utils",
        SimpleNamespace(init_device_properties_triton=lambda: None, get_vectorcore_num=lambda: cores),
    )
    name = "vllm_ascend.ops.triton.kimi_vit_rope"
    kernel_spec = importlib.util.spec_from_file_location(name, SOURCE.parent / "triton/kimi_vit_rope.py")
    kernel = importlib.util.module_from_spec(kernel_spec)
    monkeypatch.setitem(sys.modules, name, kernel)
    kernel_spec.loader.exec_module(kernel)
    return kernel


@pytest.mark.parametrize("tokens,cores", [(1, 48), (9, 48), (21904, 48), (43808, 40)])
@pytest.mark.parametrize("dtype,tile", [(torch.bfloat16, 8), (torch.float32, 4)])
def test_real_image_launch_grid_is_bounded_by_vector_cores(monkeypatch, tokens, cores, dtype, tile):
    kernel = load_kernel(monkeypatch, cores)
    launches = []

    class Capture:
        def __getitem__(self, grid):
            launches.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(kernel, "_kimi_vit_rope_kernel", Capture())
    packed = torch.empty(tokens, 3, 12, 128, dtype=dtype, device="meta")
    coeff = torch.empty(tokens, 64, device="meta")
    q, k = kernel.fused_kimi_vit_rope(packed, coeff, coeff)
    assert launches == [(min(cores, (tokens + tile - 1) // tile),)]
    assert q.is_contiguous() and k.is_contiguous()


def test_attention_preserves_packing_and_sequence_metadata(monkeypatch):
    packed = torch.randn(7, 3, 12, 128)
    freqs = torch.polar(torch.ones(7, 64), torch.randn(7, 64))
    tables = rope.prepare_kimi_vit_rope(freqs, "cache")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=rotary_primitive))
    attn = Mock(side_effect=lambda q, k, v, **kwargs: q + k + v)
    layer = SimpleNamespace(
        wqkv=lambda x: (packed.flatten(1), None),
        wo=lambda x: (x, None),
        num_attention_heads_per_partition=12,
        hidden_size_per_attention_head=128,
        attn=attn,
    )
    wrapped = rope.wrap_kimi_vit_attention(Mock(side_effect=AssertionError("unexpected fallback")))
    lengths = torch.tensor([3, 4])
    cumulative = torch.tensor([0, 3, 7])
    maximum = torch.tensor(4)
    actual = wrapped(layer, torch.empty(7, 1024), cumulative, tables, maximum, lengths)
    expected = reference(packed[:, 0], freqs) + reference(packed[:, 1], freqs) + packed[:, 2]
    torch.testing.assert_close(actual, expected.flatten(1), rtol=0, atol=0)
    assert attn.call_args.kwargs["cu_seqlens"] is cumulative
    assert attn.call_args.kwargs["max_seqlen"] is maximum
    assert attn.call_args.kwargs["sequence_lengths"] is lengths
    assert rope.wrap_kimi_vit_attention(wrapped) is wrapped


@pytest.mark.skipif(os.getenv("TRITON_INTERPRET") != "1", reason="Opt-in CPU execution of the actual Triton kernel")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,heads,dim,strided",
    [
        (0, 12, 128, False),
        (17, 12, 128, False),
        (17, 5, 128, True),
        (5, 3, 8, True),
        (2051, 1, 4, False),
    ],
)
def test_fused_kernel_interpreter_matches_complex_reference(monkeypatch, dtype, tokens, heads, dim, strided):
    if dtype == torch.bfloat16 and tokens:
        pytest.skip("Triton 3.5 CPU interpreter truncates BF16 casts; real-NPU test must verify RTNE")
    load_kernel(monkeypatch)
    torch.manual_seed(17)
    packed = torch.randn(tokens, 3, heads, dim * (2 if strided else 1), dtype=dtype)
    if strided:
        packed = packed[..., ::2]
    original = packed.clone()
    freqs = torch.polar(torch.ones(tokens, dim // 2), torch.randn(tokens, dim // 2))
    tables = rope.prepare_kimi_vit_rope(freqs, "fused")
    q, k, v = rope.apply_kimi_vit_rope(packed, tables)
    # CPU complex multiplication can contract FP32 arithmetic; the kernel
    # deliberately uses separate products, as the original NPU path does.
    torch.testing.assert_close(q, reference(packed[:, 0], freqs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k, reference(packed[:, 1], freqs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert q.is_contiguous() and k.is_contiguous()
    assert v.data_ptr() == packed[:, 2].data_ptr()
