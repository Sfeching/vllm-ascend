# SPDX-License-Identifier: Apache-2.0
"""Independent-process NPU measurements including batch table preparation."""

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "off", "cache", "fused"], default="all")
    parser.add_argument("--grids", default="1,148,148", help="T,H,W groups separated by semicolons")
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--consumer", choices=["materialize", "fia"], default="materialize")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.layers, args.repeats, args.warmups) < 1:
        parser.error("layers, repeats and warmups must be positive")
    if args.consumer == "fia" and args.dtype == "float32":
        parser.error("FIA acceptance uses BF16/FP16")
    grids = [list(map(int, group.split(","))) for group in args.grids.split(";")]
    if any(len(grid) != 3 or min(grid) < 1 for grid in grids):
        parser.error("each grid must contain positive T,H,W")
    if args.output.exists():
        parser.error("output already exists; choose a fresh artifact path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        modes = {}
        for mode in ("off", "cache", "fused"):
            child = args.output.with_name(f"{args.output.stem}-{mode}.json")
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--mode",
                    mode,
                    "--grids",
                    args.grids,
                    "--layers",
                    str(args.layers),
                    "--repeats",
                    str(args.repeats),
                    "--warmups",
                    str(args.warmups),
                    "--dtype",
                    args.dtype,
                    "--consumer",
                    args.consumer,
                    "--output",
                    str(child),
                ],
                check=True,
            )
            modes[mode] = json.loads(child.read_text())
        report = {"process_isolation": True, "modes": modes}
    else:
        report = measure(args, grids)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def measure(args, grids):
    # Parent process never initializes an NPU or retains another mode's outputs.
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.kimi_k25_vit import Rope2DPosEmbRepeated

    from vllm_ascend.ops.kimi_vit_rope import apply_kimi_vit_rope, prepare_kimi_vit_rope
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb

    torch.manual_seed(29)
    torch.set_grad_enabled(False)
    lengths = [h * w for t, h, w in grids for _ in range(t)]
    tokens = sum(lengths)
    dtype = getattr(torch, args.dtype)
    packed = torch.randn(tokens, 3, 12, 128, dtype=dtype, device="npu")
    position = Rope2DPosEmbRepeated(128, max(g[1] for g in grids), max(g[2] for g in grids))
    freqs = position.get_freqs_cis(grids, torch.device("cpu")).npu()
    cumulative = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to("npu", dtype=torch.int32)
    maximum = torch.tensor(max(lengths))
    with set_current_vllm_config(VllmConfig()):
        baseline = AscendApplyRotaryEmb(enforce_enable=True, is_neox_style=False, enable_fp32_compute=True)
        attention = AscendMMEncoderAttention(num_heads=12, head_size=128) if args.consumer == "fia" else None

    def layer(mode, tables):
        if mode == "off":
            cos, sin = freqs.real.contiguous(), freqs.imag.contiguous()
            q = baseline.forward_oot(packed[:, 0], cos, sin)
            k = baseline.forward_oot(packed[:, 1], cos, sin)
            v = packed[:, 2]
        else:
            q, k, v = apply_kimi_vit_rope(packed, tables)
        if attention is not None:
            return attention.forward_oot(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                cu_seqlens=cumulative,
                max_seqlen=maximum,
            )
        # Same consumer-side materialization as FIA; V remains outside fusion.
        return q.contiguous(), k.contiguous(), v.contiguous()

    def batch():
        tables = prepare_kimi_vit_rope(freqs, args.mode) if args.mode != "off" else None
        for _ in range(args.layers):
            output = layer(args.mode, tables)
            del output  # No previous layer's Q/K/V kept alive across allocations.

    expected = layer("off", None)
    tables = prepare_kimi_vit_rope(freqs, args.mode) if args.mode != "off" else None
    actual = layer(args.mode, tables)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    for result, reference in zip(actual_items, expected_items, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
    del expected, actual, expected_items, actual_items, result, reference, tables
    for _ in range(args.warmups):
        batch()
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    allocated_before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        batch()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mode": args.mode,
        "grids": grids,
        "tokens": tokens,
        "layers": args.layers,
        "dtype": args.dtype,
        "consumer": args.consumer,
        "correctness_passed": True,
        "scope": "one table preparation + repeated RoPE and selected consumer; excludes QKV projection/MLP",
        "batch_ms_median": statistics.median(samples),
        "batch_ms_p95": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "batch_ms_samples": samples,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_increment_bytes": torch.npu.max_memory_allocated() - allocated_before,
        "allocated_after_bytes": torch.npu.memory_allocated(),
        "torch_version": torch.__version__,
    }


if __name__ == "__main__":
    main()
