# SPDX-License-Identifier: Apache-2.0
"""Opt-in MoonViT CP groups, global-position sharding and complete encoder path."""

import itertools
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ViTCPLayout:
    lengths: tuple[int, ...]
    cp_size: int
    rank: int
    ranges: tuple[tuple[tuple[int, int], ...], ...]
    padded_lengths: tuple[int, ...]
    local_lengths: tuple[int, ...]
    indices: tuple[int, ...]
    q_ends: tuple[int, ...]
    kv_ends: tuple[int, ...]


def make_layout(grids, cp_size, rank):
    if cp_size not in (1, 2, 4) or not 0 <= rank < cp_size:
        raise ValueError("MoonViT CP supports sizes 1/2/4 and a valid rank")
    if not grids or any(len(grid) != 3 or min(grid) <= 0 for grid in grids):
        raise ValueError("Each grid must contain positive T,H,W")
    lengths = tuple(t * h * w for t, h, w in grids)
    if min(lengths) < cp_size:
        raise ValueError("Each image must give all CP ranks a nonempty shard")
    ranges = tuple(tuple((n * r // cp_size, n * (r + 1) // cp_size) for r in range(cp_size)) for n in lengths)
    local_lengths = tuple(parts[rank][1] - parts[rank][0] for parts in ranges)
    indices = []
    offset = 0
    for n, parts in zip(lengths, ranges, strict=True):
        start, end = parts[rank]
        indices.extend(range(offset + start, offset + end))
        offset += n
    return ViTCPLayout(
        lengths,
        cp_size,
        rank,
        ranges,
        tuple((n + cp_size - 1) // cp_size for n in lengths),
        local_lengths,
        tuple(indices),
        tuple(itertools.accumulate(local_lengths)),
        tuple(itertools.accumulate(lengths)),
    )


def shard_tokens(full, layout):
    parts = []
    offset = 0
    for n, ranges in zip(layout.lengths, layout.ranges, strict=True):
        start, end = ranges[layout.rank]
        parts.append(full[offset + start : offset + end])
        offset += n
    return torch.cat(parts, dim=0)


def pad_tokens(local, layout):
    output = local.new_zeros((sum(layout.padded_lengths), *local.shape[1:]))
    source = destination = 0
    for count, padded in zip(layout.local_lengths, layout.padded_lengths, strict=True):
        output[destination : destination + count].copy_(local[source : source + count])
        source += count
        destination += padded
    return output


def restore_tokens(gathered, layout):
    parts = []
    offset = 0
    for ranges, padded in zip(layout.ranges, layout.padded_lengths, strict=True):
        for rank, (start, end) in enumerate(ranges):
            parts.append(gathered[rank, offset : offset + end - start])
        offset += padded
    return torch.cat(parts, dim=0)


def pack_kv(key, value, layout):
    return torch.stack((pad_tokens(key, layout), pad_tokens(value, layout)), dim=0)


def restore_kv(gathered, layout):
    # Restoring with the token axis first produces one contiguous K/V buffer.
    parts = []
    offset = 0
    for ranges, padded in zip(layout.ranges, layout.padded_lengths, strict=True):
        for rank, (start, end) in enumerate(ranges):
            parts.append(gathered[rank, :, offset : offset + end - start])
        offset += padded
    output = torch.cat(parts, dim=1)
    return output[0], output[1]


def assign_images(lengths, groups):
    if groups <= 0 or any(n <= 0 for n in lengths):
        raise ValueError("Positive group count and image lengths are required")
    assignment = [[] for _ in range(groups)]
    loads = [0] * groups
    for image in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        group = min(range(groups), key=lambda g: (loads[g], g))
        assignment[group].append(image)
        loads[group] += lengths[image]
    return tuple(tuple(images) for images in assignment)


def cp_group_ranks(tp_groups, cp_size):
    if cp_size not in (1, 2, 4) or any(len(ranks) % cp_size for ranks in tp_groups):
        raise ValueError("Each TP group must divide evenly into CP1/2/4 groups")
    return [ranks[start : start + cp_size] for ranks in tp_groups for start in range(0, len(ranks), cp_size)]


def scope(name):
    return torch.profiler.record_function(name) if torch.autograd._profiler_enabled() else nullcontext()


@dataclass
class ViTCPContext:
    group: object
    tp_group: object
    copy_v: bool = False
    linear_batch_mode: str = "replicated"
    batch_count: int = 0
    image_count: int = 0
    patch_count: int = 0
    layer_count: int = 0
    fallback_count: int = 0
    recent_batches: list = field(default_factory=list)

    @property
    def rank(self):
        return self.group.rank_in_group

    @property
    def size(self):
        return self.group.world_size

    @property
    def group_index(self):
        return self.tp_group.rank_in_group // self.size

    def all_gather(self, tensor):
        if self.size == 1:
            return tensor.unsqueeze(0)
        output = torch.empty((self.size * tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(output, tensor, group=self.group.device_group)
        return output.view(self.size, *tensor.shape)

    def broadcast(self, tensor):
        return self.group.broadcast(tensor, src=0)

    def identity(self):
        return {
            "cp_size": self.size,
            "cp_rank": self.rank,
            "cp_group": self.group_index,
            "cp_global_ranks": self.group.ranks,
            "copy_v": self.copy_v,
            "linear_batch_mode": self.linear_batch_mode,
            "batch_count": self.batch_count,
            "image_count": self.image_count,
            "patch_count": self.patch_count,
            "layer_count": self.layer_count,
            "fallback_count": self.fallback_count,
            "recent_batches": list(self.recent_batches),
        }


def create_context(cp_size, copy_v=False):
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import GroupCoordinator

    tp = get_tp_group()
    groups = cp_group_ranks(tp.group_ranks, cp_size)
    group = GroupCoordinator(
        groups,
        tp.local_rank,
        dist.get_backend(tp.device_group),
        use_device_communicator=False,
        use_message_queue_broadcaster=False,
        group_name="kimi_vit_cp",
    )
    return ViTCPContext(group, tp, copy_v)


def encoder_forward_sharded_linears(encoder, full_hidden, grids, context, rope_mode="fused", rotate=None):
    """Experimental full sharding; real BF16 weights failed the feature budget."""
    context.linear_batch_mode = "sharded"
    from vllm_ascend.ops.kimi_vit_rope import KimiViTRopeTables, apply_kimi_vit_rope, prepare_kimi_vit_rope

    layout = make_layout(grids, context.size, context.rank)
    hidden = shard_tokens(full_hidden, layout)
    frequencies = encoder.rope_2d.get_freqs_cis(grids, device=torch.device("cpu"))
    cpu_tables = prepare_kimi_vit_rope(shard_tokens(frequencies, layout), rope_mode)
    tables = KimiViTRopeTables(
        cpu_tables.cos.to(hidden.device),
        cpu_tables.sin.to(hidden.device),
        rope_mode,
        hidden.shape[0],
        cpu_tables.head_dim,
    )
    del frequencies, cpu_tables, full_hidden
    if rotate is None:
        if context.copy_v:
            if rope_mode != "fused":
                raise ValueError("V fusion requires fused RoPE")
            from vllm_ascend.ops.triton.kimi_vit_rope_qkv import fused_kimi_vit_qkv

            def rotate(packed, coeff):
                return fused_kimi_vit_qkv(packed, coeff.cos, coeff.sin)
        else:
            rotate = apply_kimi_vit_rope

    prefix = f"kimi.vit.cp{context.size}.g{context.group_index}"
    for index, block in enumerate(encoder.blocks):
        with scope(f"{prefix}.layer{index}"):
            residual = hidden
            with scope("kimi.vit.norm_qkv_rope"):
                normalized = block.norm0(hidden)
                qkv, _ = block.wqkv(normalized)
                packed = qkv.view(
                    hidden.shape[0], 3, block.num_attention_heads_per_partition, block.hidden_size_per_attention_head
                )
                q, k, v = rotate(packed, tables)
            with scope("kimi.vit.kv_pack"):
                send = pack_kv(k, v, layout)
            with scope("kimi.vit.kv_allgather"):
                gathered = context.all_gather(send)
            with scope("kimi.vit.kv_restore"):
                key, value = restore_kv(gathered, layout)
            with scope("kimi.vit.fia"):
                attended = block.attn._run_vit_fia(q, key, value, list(layout.q_ends), list(layout.kv_ends))
            with scope("kimi.vit.o_projection"):
                projected, _ = block.wo(attended.reshape(hidden.shape[0], -1))
                hidden = residual + projected
            with scope("kimi.vit.norm_mlp"):
                hidden = hidden + block.mlp(block.norm1(hidden))
            del normalized, qkv, packed, q, k, v, send, gathered, key, value, attended, projected, residual
            context.layer_count += 1
    hidden = encoder.final_layernorm(hidden)
    with scope("kimi.vit.hidden_allgather"):
        full = restore_tokens(context.all_gather(pad_tokens(hidden, layout)), layout)
    return full


def encoder_forward(encoder, full_hidden, grids, context, rope_mode="fused", rotate=None):
    """Shard FIA queries while preserving the original dense GEMM dimensions.

    BF16 MLP rounding changed with local GEMM row counts and accumulated across
    27 real-weight layers. Replicate the dense path and gather attention output;
    complete K/V is already available, so no K/V exchange is needed.
    """
    from vllm_ascend.ops.kimi_vit_rope import KimiViTRopeTables, apply_kimi_vit_rope, prepare_kimi_vit_rope

    context.linear_batch_mode = "replicated"
    layout = make_layout(grids, context.size, context.rank)
    hidden = full_hidden
    frequencies = encoder.rope_2d.get_freqs_cis(grids, device=torch.device("cpu"))
    prepared = prepare_kimi_vit_rope(frequencies, rope_mode)
    tables = KimiViTRopeTables(
        prepared.cos.to(hidden.device), prepared.sin.to(hidden.device),
        rope_mode, hidden.shape[0], prepared.head_dim,
    )
    del frequencies, prepared, full_hidden
    if rotate is None:
        if context.copy_v:
            if rope_mode != "fused":
                raise ValueError("V fusion requires fused RoPE")
            from vllm_ascend.ops.triton.kimi_vit_rope_qkv import fused_kimi_vit_qkv

            def rotate(packed, coeff):
                return fused_kimi_vit_qkv(packed, coeff.cos, coeff.sin)
        else:
            rotate = apply_kimi_vit_rope

    prefix = f"kimi.vit.cp{context.size}.g{context.group_index}"
    for index, block in enumerate(encoder.blocks):
        with scope(f"{prefix}.layer{index}"):
            residual = hidden
            with scope("kimi.vit.norm_qkv_rope"):
                normalized = block.norm0(hidden)
                qkv, _ = block.wqkv(normalized)
                packed = qkv.view(
                    hidden.shape[0], 3, block.num_attention_heads_per_partition,
                    block.hidden_size_per_attention_head,
                )
                full_query, key, value = rotate(packed, tables)
                query = shard_tokens(full_query, layout)
            with scope("kimi.vit.fia"):
                local_attention = block.attn._run_vit_fia(
                    query, key, value, list(layout.q_ends), list(layout.kv_ends)
                )
            with scope("kimi.vit.attention_pack"):
                send = pad_tokens(local_attention, layout)
            with scope("kimi.vit.attention_allgather"):
                gathered = context.all_gather(send)
            with scope("kimi.vit.attention_restore"):
                attended = restore_tokens(gathered, layout)
            with scope("kimi.vit.o_projection"):
                projected, _ = block.wo(attended.reshape(hidden.shape[0], -1))
                hidden = residual + projected
            with scope("kimi.vit.norm_mlp"):
                hidden = hidden + block.mlp(block.norm1(hidden))
            del normalized, qkv, packed, full_query, query, key, value
            del local_attention, send, gathered, attended, projected, residual
            context.layer_count += 1
    return encoder.final_layernorm(hidden)


def vision_forward(vision_tower, pixels, grids, mm_projector, context, rope_mode):
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.models.kimi_k25_vit import mm_projector_forward, tpool_patch_merger

    lengths = [t * h * w for t, h, w in grids]
    groups = context.tp_group.world_size // context.size
    assignment = assign_images(lengths, groups)
    image_ids = assignment[context.group_index]
    offsets = [0, *itertools.accumulate(lengths)]
    local_grids = [grids[image] for image in image_ids]
    merge_h, merge_w = vision_tower.merge_kernel_size
    output_lengths = [(h // merge_h) * (w // merge_w) for t, h, w in grids]
    max_outputs = max(sum(output_lengths[i] for i in images) for images in assignment)
    hidden_dim = vision_tower.encoder.blocks[0].hidden_dim
    feature_dtype = next(vision_tower.encoder.parameters()).dtype
    merged = pixels.new_zeros((max_outputs, merge_h * merge_w, hidden_dim), dtype=feature_dtype)
    if image_ids:
        fallback = any(t * h * w < context.size for t, h, w in local_grids)
        if fallback:
            if context.rank == 0:
                local_pixels = torch.cat([pixels[offsets[i] : offsets[i + 1]] for i in image_ids], dim=0)
                outputs = vision_tower(local_pixels, local_grids)
                current = torch.cat(outputs, dim=0)
                merged[: current.shape[0]].copy_(current)
            context.fallback_count += 1
        else:
            full_tokens = sum(lengths[i] for i in image_ids)
            if context.rank == 0:
                with scope("kimi.vit.patch_embed"):
                    local_pixels = torch.cat([pixels[offsets[i] : offsets[i + 1]] for i in image_ids], dim=0)
                    hidden = vision_tower.patch_embed(local_pixels, local_grids)
            else:
                hidden = pixels.new_empty((full_tokens, hidden_dim), dtype=feature_dtype)
            with scope("kimi.vit.input_broadcast"):
                context.broadcast(hidden)
            full = encoder_forward(vision_tower.encoder, hidden, local_grids, context, rope_mode)
            if context.rank == 0:
                with scope("kimi.vit.patch_merger"):
                    current = torch.cat(tpool_patch_merger(full, local_grids, vision_tower.merge_kernel_size), dim=0)
                    merged[: current.shape[0]].copy_(current)
            context.batch_count += 1
            context.image_count += len(image_ids)
            context.patch_count += full_tokens
            context.recent_batches.append(
                {
                    "image_ids": list(image_ids),
                    "grids": local_grids,
                    "patches": full_tokens,
                    "local_patches": sum(make_layout(local_grids, context.size, context.rank).local_lengths),
                }
            )
            context.recent_batches[:] = context.recent_batches[-32:]
    # Only a leader contributes features. All TP ranks participate, including
    # idle groups, and the original projector remains on the original path.
    with scope("kimi.vit.features_allgather"):
        gathered = tensor_model_parallel_all_gather(merged, dim=0)
    reordered = [None] * len(grids)
    for group, images in enumerate(assignment):
        offset = group * context.size * max_outputs
        for image in images:
            reordered[image] = gathered[offset : offset + output_lengths[image]]
            offset += output_lengths[image]
    with scope("kimi.vit.projector"):
        return list(mm_projector_forward(mm_projector, reordered))


def wrap_vision_forward(forward):
    if getattr(forward, "_ascend_kimi_vit_cp_wrapped", False):
        return forward

    @wraps(forward)
    def wrapped(vision_tower, pixel_values, grid_thw, mm_projector, use_data_parallel):
        from vllm_ascend import envs

        context = getattr(vision_tower, "_ascend_cp_context", None)
        with scope("kimi.vit.forward"):
            if context is None:
                return forward(vision_tower, pixel_values, grid_thw, mm_projector, use_data_parallel)
            if not use_data_parallel:
                raise ValueError("MoonViT CP requires replicated vision weights (mm_encoder_tp_mode=data)")
            grids = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            if not grids:
                return []
            return vision_forward(
                vision_tower, pixel_values, grids, mm_projector, context, envs.VLLM_ASCEND_KIMI_VIT_ROPE_MODE
            )

    wrapped._ascend_kimi_vit_cp_wrapped = True
    return wrapped
