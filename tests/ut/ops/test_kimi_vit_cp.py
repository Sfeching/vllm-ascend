# SPDX-License-Identifier: Apache-2.0
"""Image assignment, global-position sharding and CP data restoration."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_cp():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_vit_cp.py"
    assert path.exists(), "Complete-service CP implementation is missing"
    spec = importlib.util.spec_from_file_location("cp_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_unequal_images_and_video_restore_without_padding_or_reordering(cp_size):
    cp = load_cp()
    grids = [[1, 5, 7], [2, 6, 4]]
    torch.manual_seed(31)
    full = torch.randn(83, 3, 12, 8)
    # This is also the global position identity: no restarting at local zero.
    global_positions = torch.arange(83)
    shards, kv = [], []
    for rank in range(cp_size):
        layout = cp.make_layout(grids, cp_size, rank)
        local = cp.shard_tokens(full, layout)
        shards.append(cp.pad_tokens(local, layout))
        kv.append(cp.pack_kv(local[:, 1], local[:, 2], layout))
        actual_positions = cp.shard_tokens(global_positions, layout)
        assert actual_positions.tolist() == list(layout.indices)
        assert layout.q_ends[-1] == local.shape[0]
        assert layout.kv_ends == (35, 83)
    for rank in range(cp_size):
        layout = cp.make_layout(grids, cp_size, rank)
        restored = cp.restore_tokens(torch.stack(shards), layout)
        key, value = cp.restore_kv(torch.stack(kv), layout)
        torch.testing.assert_close(restored, full, rtol=0, atol=0)
        torch.testing.assert_close(key, full[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(value, full[:, 2], rtol=0, atol=0)
        assert key.is_contiguous() and value.is_contiguous()


def test_assignment_is_consistent_patch_balanced_and_keeps_idle_groups():
    cp = load_cp()
    assert cp.assign_images([100, 90, 60, 50, 30], 4) == ((0,), (1,), (2,), (3, 4))
    assert cp.assign_images([100, 90], 4) == ((0,), (1,), (), ())
    assert cp.assign_images([], 4) == ((), (), (), ())
    assert cp.cp_group_ranks([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


@pytest.mark.parametrize(
    "grids,size,rank", [([[1, 0, 8]], 4, 0), ([[1, 1, 3]], 4, 0), ([[1, 8, 8]], 3, 0), ([[1, 8, 8]], 4, 4)]
)
def test_layout_rejects_invalid_or_empty_shards(grids, size, rank):
    cp = load_cp()
    with pytest.raises(ValueError):
        cp.make_layout(grids, size, rank)


def test_partial_tp_groups_are_rejected():
    cp = load_cp()
    with pytest.raises(ValueError):
        cp.cp_group_ranks([[0, 1, 2, 3, 4]], 4)
