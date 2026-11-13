# SPDX-License-Identifier: Apache-2.0
"""Local ViT CP prototype: global-position RoPE, KV all-gather and full-image FIA.

Run with torchrun --nproc-per-node=1/2/4. This does not modify a model or service.
"""

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path


def make_layout(lengths, world, rank):
    if world < 1 or not 0 <= rank < world or not lengths or min(lengths) < world:
        raise ValueError("Each image must provide a nonempty shard to every valid CP rank")
    ranges = [[(n * r // world, n * (r + 1) // world) for r in range(world)] for n in lengths]
    local_lengths = [parts[rank][1] - parts[rank][0] for parts in ranges]
    return {
        "world": world,
        "rank": rank,
        "lengths": lengths,
        "ranges": ranges,
        "local_lengths": local_lengths,
        "padded_lengths": [(n + world - 1) // world for n in lengths],
        "q_ends": list(itertools.accumulate(local_lengths)),
        "kv_ends": list(itertools.accumulate(lengths)),
    }


def local_indices(layout):
    import torch

    offset = 0
    parts = []
    for n, ranges in zip(layout["lengths"], layout["ranges"], strict=True):
        start, end = ranges[layout["rank"]]
        parts.append(torch.arange(offset + start, offset + end))
        offset += n
    return torch.cat(parts)


def pack_kv(key, value, layout):
    import torch

    packed = torch.zeros((2, sum(layout["padded_lengths"]), *key.shape[1:]), dtype=key.dtype, device=key.device)
    source = destination = 0
    for count, padded in zip(layout["local_lengths"], layout["padded_lengths"], strict=True):
        packed[0, destination : destination + count].copy_(key[source : source + count])
        packed[1, destination : destination + count].copy_(value[source : source + count])
        source += count
        destination += padded
    return packed


def restore_kv(gathered, layout):
    import torch

    parts = []
    offset = 0
    for ranges, padded in zip(layout["ranges"], layout["padded_lengths"], strict=True):
        for rank, (start, end) in enumerate(ranges):
            parts.append(gathered[rank, :, offset : offset + end - start])
        offset += padded
    full = torch.cat(parts, dim=1)
    return full[0], full[1]


def install_kernel(path):
    name = "vllm_ascend.ops.triton.kimi_vit_rope"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def measure_case(args, grids, rank, world, correctness_only=False):
    # Imports remain local so layout tests and launcher parents need no NPU.
    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.kimi_k25_vit import Rope2DPosEmbRepeated

    from vllm_ascend.ops.kimi_vit_rope import apply_kimi_vit_rope, prepare_kimi_vit_rope
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb

    lengths = [h * w for t, h, w in grids for _ in range(t)]
    layout = make_layout(lengths, world, rank)
    index = local_indices(layout)
    torch.manual_seed(29)
    full_cpu = torch.randn(sum(lengths), 3, 12, 128, dtype=torch.bfloat16)
    packed = full_cpu[index].contiguous().npu()
    position = Rope2DPosEmbRepeated(128, max(g[1] for g in grids), max(g[2] for g in grids))
    freqs = position.get_freqs_cis(grids, torch.device("cpu"))
    local_freqs = freqs[index].contiguous().npu()
    tables = prepare_kimi_vit_rope(local_freqs, "fused")
    with set_current_vllm_config(VllmConfig()):
        attention = AscendMMEncoderAttention(num_heads=12, head_size=128)
        baseline = AscendApplyRotaryEmb(enforce_enable=True, is_neox_style=False, enable_fp32_compute=True)

    def exchange(key, value):
        send = pack_kv(key, value, layout)
        if world == 1:
            gathered = send.unsqueeze(0)
        else:
            gathered = torch.empty((world * 2, *send.shape[1:]), dtype=send.dtype, device=send.device)
            dist.all_gather_into_tensor(gathered, send)
            gathered = gathered.view(world, *send.shape)
        return restore_kv(gathered, layout)

    full = full_cpu.npu()
    cos, sin = freqs.real.contiguous().npu(), freqs.imag.contiguous().npu()
    expected_q = baseline.forward_oot(full[:, 0], cos, sin)
    expected_k = baseline.forward_oot(full[:, 1], cos, sin)
    q, k, v = apply_kimi_vit_rope(packed, tables)
    key, value = exchange(k, v)
    torch.testing.assert_close(q.cpu(), expected_q.cpu()[index], rtol=0, atol=0)
    torch.testing.assert_close(key, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(value, full[:, 2], rtol=0, atol=0)
    reference = (
        attention._run_vit_fia(expected_q, expected_k, full[:, 2], layout["kv_ends"], layout["kv_ends"])
        .cpu()[index]
        .float()
    )
    actual = attention._run_vit_fia(q, key, value, layout["q_ends"], layout["kv_ends"]).cpu().float()
    difference = actual - reference
    errors = {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "exact_elements_fraction": (actual == reference).float().mean().item(),
        "reference_rms": reference.square().mean().sqrt().item(),
        "rtol": 0.01,
        "atol": 0.0005,
    }
    torch.testing.assert_close(actual, reference, rtol=errors["rtol"], atol=errors["atol"])
    if errors["rms"] > 0.0001:
        raise AssertionError(f"FIA RMS exceeds 1e-4: {errors}")
    del full_cpu, full, cos, sin, expected_q, expected_k, q, k, v, key, value, reference, actual, difference
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    report = {"grids": grids, "layout": layout, "qkv_kv_exact": True, "fia_errors": errors}
    if correctness_only:
        return report

    def layer(batch_tables):
        q, k, v = apply_kimi_vit_rope(packed, batch_tables)
        key, value = exchange(k, v)
        return attention._run_vit_fia(q, key, value, layout["q_ends"], layout["kv_ends"])

    def batch():
        batch_tables = prepare_kimi_vit_rope(local_freqs, "fused")
        for _ in range(args.layers):
            output = layer(batch_tables)
            del output

    for _ in range(args.warmups):
        batch()
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        if world > 1:
            dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        batch()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    report.update(
        batch_ms_samples=samples,
        batch_ms_median=statistics.median(samples),
        allocated_before_bytes=before,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        peak_increment_bytes=torch.npu.max_memory_allocated() - before,
    )
    # Separate diagnostic passes: per-stage synchronization is excluded from
    # the above unsplit timing, and no component medians are summed as total.
    stage_samples = {name: [] for name in ("rope", "kv_pack", "all_gather", "kv_reorder", "fia")}
    for _ in range(args.repeats):
        if world > 1:
            dist.barrier()

        def timed(name, operation, *inputs):
            torch.npu.synchronize()
            start = time.perf_counter()
            result = operation(*inputs)
            torch.npu.synchronize()
            stage_samples[name].append((time.perf_counter() - start) * 1000)
            return result

        q, k, v = timed("rope", apply_kimi_vit_rope, packed, tables)
        send = timed("kv_pack", pack_kv, k, v, layout)
        if world > 1:
            gathered = torch.empty((world * 2, *send.shape[1:]), dtype=send.dtype, device=send.device)
            timed("all_gather", dist.all_gather_into_tensor, gathered, send)
            gathered = gathered.view(world, *send.shape)
        else:
            gathered = send.unsqueeze(0)
            stage_samples["all_gather"].append(0.0)
        key, value = timed("kv_reorder", restore_kv, gathered, layout)
        output = timed("fia", attention._run_vit_fia, q, key, value, layout["q_ends"], layout["kv_ends"])
        del q, k, v, send, gathered, key, value, output
    report["stage_ms_samples"] = stage_samples
    report["stage_ms_median"] = {key: statistics.median(values) for key, values in stage_samples.items()}
    report["kv_logical_bytes_per_layer"] = sum(lengths) * 2 * 12 * 128 * 2
    report["kv_send_bytes_per_rank"] = sum(layout["padded_lengths"]) * 2 * 12 * 128 * 2
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.layers, args.warmups, args.repeats) < 1:
        parser.error("layers, warmups and repeats must be positive")
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.set_grad_enabled(False)
    if world > 1:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    args.output.mkdir(parents=True, exist_ok=True)
    rank_file = args.output / f"cp{world}-rank{rank}.json"
    if rank_file.exists():
        raise FileExistsError(rank_file)
    install_kernel(args.kernel)
    report = {
        "rank": rank,
        "world": world,
        "layers": args.layers,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "kernel_sha256": hashlib.sha256(args.kernel.read_bytes()).hexdigest(),
        "scope": "batch coefficient preparation + 27 repeated RoPE/KV pack/all-gather/reorder/FIA; "
        "excludes QKV/O projection, MLP, final output gather and complete model",
        "timing": "synchronized wall time; total uses max rank per aligned sample; stages are separate diagnostics",
    }
    report["small_unequal_images"] = measure_case(args, [[1, 5, 7], [1, 7, 9]], rank, world, True)
    print(f"CP{world} rank{rank}: unequal-image correctness passed", flush=True)
    report["cases"] = []
    for images in (1, 2):
        case = measure_case(args, [[1, 148, 148]] * images, rank, world)
        report["cases"].append(case)
        print(f"CP{world} rank{rank} images{images}: {case['batch_ms_median']:.3f} ms", flush=True)
        gc.collect()
        torch.npu.empty_cache()
    rank_file.write_text(json.dumps(report, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    if rank == 0:
        ranks = [json.loads((args.output / f"cp{world}-rank{r}.json").read_text()) for r in range(world)]
        summary = {"world": world, "rank_reports": [f"cp{world}-rank{r}.json" for r in range(world)], "cases": []}
        for case_index in (0, 1):
            cases = [r["cases"][case_index] for r in ranks]
            maxima = [max(values) for values in zip(*(c["batch_ms_samples"] for c in cases), strict=True)]
            summary["cases"].append(
                {
                    "images": case_index + 1,
                    "batch_max_rank_ms_samples": maxima,
                    "batch_max_rank_ms_median": statistics.median(maxima),
                    "stage_max_rank_median_ms": {
                        key: max(c["stage_ms_median"][key] for c in cases) for key in cases[0]["stage_ms_median"]
                    },
                    "fia_max_abs": max(c["fia_errors"]["max_abs"] for c in cases),
                    "fia_max_rms": max(c["fia_errors"]["rms"] for c in cases),
                    "peak_increment_max_rank_bytes": max(c["peak_increment_bytes"] for c in cases),
                }
            )
        (args.output / f"cp{world}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
