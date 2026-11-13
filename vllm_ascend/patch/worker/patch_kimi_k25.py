#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.kimi_k25 as kimi_model
import vllm.model_executor.models.kimi_k25_vit as kimi_vision
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
    MoonViT3dEncoder,
    MoonViT3dPretrainedModel,
    MoonViTEncoderLayer,
    get_rope_shape_decorate,
)

from vllm_ascend import envs
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.kimi_vit_cp import create_context, wrap_vision_forward
from vllm_ascend.ops.kimi_vit_rope import wrap_kimi_vit_attention, wrap_kimi_vit_encoder_forward


def _kimi_vit_rope_mode():
    mode = envs.VLLM_ASCEND_KIMI_VIT_ROPE_MODE
    if mode != "off" and not get_current_hardware_profile().supports(HardwareCapability.FUSED_ROTARY_MUL_INTERLEAVE):
        raise NotImplementedError("Experimental Kimi ViT RoPE requires an interleave-capable Ascend device")
    return mode


MoonViT3dEncoder.forward = wrap_kimi_vit_encoder_forward(MoonViT3dEncoder.forward, _kimi_vit_rope_mode)
MoonViTEncoderLayer.attention_qkvpacked = wrap_kimi_vit_attention(MoonViTEncoderLayer.attention_qkvpacked)

_original_moonvit_init = MoonViT3dPretrainedModel.__init__


def _moonvit_init_with_cp(self, *args, **kwargs):
    _original_moonvit_init(self, *args, **kwargs)
    size = envs.VLLM_ASCEND_KIMI_VIT_CP_SIZE
    if size not in (1, 2, 4):
        raise ValueError("VLLM_ASCEND_KIMI_VIT_CP_SIZE must be 1, 2 or 4")
    if size > 1:
        if not self.encoder.blocks[0].use_data_parallel:
            raise ValueError("MoonViT CP requires replicated visual weights")
        if _kimi_vit_rope_mode() not in ("cache", "fused"):
            raise ValueError("MoonViT CP requires batch-scoped RoPE (cache or fused)")
        self._ascend_cp_context = create_context(size, envs.VLLM_ASCEND_KIMI_VIT_COPY_V)


MoonViT3dPretrainedModel.__init__ = _moonvit_init_with_cp
kimi_vision.vision_tower_forward = wrap_vision_forward(kimi_vision.vision_tower_forward)
# Upstream imported the function by value; update that binding too.
kimi_model.vision_tower_forward = kimi_vision.vision_tower_forward


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


class AscendLearnable2DInterpPosEmbDivided_fixed(nn.Module):
    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor | list) -> torch.Tensor:
        pos_embs = []
        if isinstance(grid_thws, torch.Tensor):
            grid_list = grid_thws.tolist()
        else:
            grid_list = grid_thws

        for t, h, w in grid_list:
            assert t <= self.num_frames, (
                f"[vllm-ascend/patch_kimi_k25] Invalid frame count. t={t}, num_frames={self.num_frames}"
            )
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


Learnable2DInterpPosEmbDivided_fixed.forward = AscendLearnable2DInterpPosEmbDivided_fixed.forward


# Patch MoonViT3dPretrainedModel.to() to ignore the `dtype` argument.
# When KimiK25ForConditionalGeneration.__init__ calls:
#     self.vision_tower = self.vision_tower.to(device=..., dtype=model_config.dtype)
# the `dtype=model_config.dtype` (e.g. bf16) would overwrite the fp8 parameters
# created by the Ascend quantization scheme, causing a dtype mismatch later
# in weight_loader when the checkpoint's fp8 weights are loaded.
if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
    _original_moonvit_to = MoonViT3dPretrainedModel.to

    def _patched_moonvit_to(self, *args, **kwargs):
        # Filter out dtype from positional arguments and remove from kwargs
        # to prevent overriding quantized weight dtypes on A5.
        new_args = tuple(a for a in args if not isinstance(a, torch.dtype))
        kwargs.pop("dtype", None)
        return _original_moonvit_to(self, *new_args, **kwargs)

    MoonViT3dPretrainedModel.to = _patched_moonvit_to
