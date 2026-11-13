# SPDX-License-Identifier: Apache-2.0
"""CP must preserve global positions, image isolation and full attention."""

import importlib.util
from pathlib import Path

import pytest
import torch


def load_benchmark():
    source = Path(__file__).resolve().parents[3] / "benchmarks/kimi_vit_cp.py"
    assert source.exists(), "The reusable ViT CP benchmark has not been implemented"
    spec = importlib.util.spec_from_file_location("kimi_vit_cp_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("world", [1, 2, 4])
def test_cp_restores_unequal_images_and_preserves_full_attention(world):
    bench = load_benchmark()
    torch.manual_seed(17)
    lengths = [15, 28]
    full = torch.randn(sum(lengths), 3, 3, 8)
    # Position-dependent Q/K makes restarting positions on each shard fail.
    positions = torch.arange(sum(lengths)).float().reshape(-1, 1, 1)
    full[:, 0] *= torch.cos(positions)
    full[:, 1] *= torch.sin(positions)
    sends, indices = [], []
    for rank in range(world):
        layout = bench.make_layout(lengths, world, rank)
        index = bench.local_indices(layout)
        indices.append(index)
        local = full[index]
        sends.append(bench.pack_kv(local[:, 1], local[:, 2], layout))
    for rank in range(world):
        layout = bench.make_layout(lengths, world, rank)
        k, v = bench.restore_kv(torch.stack(sends), layout)
        torch.testing.assert_close(k, full[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(v, full[:, 2], rtol=0, atol=0)
        assert k.is_contiguous() and v.is_contiguous()
        local_offset = global_offset = 0
        for image, length in enumerate(lengths):
            count = layout["local_lengths"][image]
            q = full[indices[rank][local_offset : local_offset + count], 0]
            full_q = full[global_offset : global_offset + length, 0]
            key = full[global_offset : global_offset + length, 1]
            value = full[global_offset : global_offset + length, 2]

            def attention(query, key=key, value=value):
                return torch.einsum(
                    "hij,jhd->ihd",
                    torch.einsum("ihd,jhd->hij", query, key).mul(8**-0.5).softmax(-1),
                    value,
                )

            expected = attention(full_q)[layout["ranges"][image][rank][0] : layout["ranges"][image][rank][1]]
            torch.testing.assert_close(attention(q), expected)
            local_offset += count
            global_offset += length
        assert layout["q_ends"][-1] == len(indices[rank])
        assert layout["kv_ends"] == [15, 43]
    assert sorted(torch.cat(indices).tolist()) == list(range(sum(lengths)))


@pytest.mark.parametrize("lengths,world,rank", [([0], 2, 0), ([3], 4, 0), ([8], 0, 0), ([8], 2, 2)])
def test_cp_rejects_empty_shards_and_invalid_ranks(lengths, world, rank):
    bench = load_benchmark()
    with pytest.raises(ValueError):
        bench.make_layout(lengths, world, rank)
