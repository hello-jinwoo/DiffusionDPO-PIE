#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import io
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms import functional as TF
from utils.training_progress import create_progress_bar
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.utils import ContextManagers

if not hasattr(torch.library, "custom_op"):
    # Allow importing latest diffusers on torch versions without torch.library.custom_op/ register_fake
    def _torch_library_custom_op_stub(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    torch.library.custom_op = _torch_library_custom_op_stub

if not hasattr(torch.library, "register_fake"):
    def _torch_library_register_fake_stub(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    torch.library.register_fake = _torch_library_register_fake_stub

if not hasattr(torch.nn, "RMSNorm"):
    # Basic RMSNorm fallback for torch versions prior to 2.3
    class _TorchFallbackRMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            dims = tuple(range(-len(self.normalized_shape), 0))
            scale = x.pow(2).mean(dim=dims, keepdim=True).add(self.eps).rsqrt()
            out = x * scale
            if self.elementwise_affine:
                view_shape = [1] * (out.dim() - len(self.normalized_shape)) + list(self.normalized_shape)
                out = out * self.weight.view(*view_shape)
            return out

    torch.nn.RMSNorm = _TorchFallbackRMSNorm

# Accept newer diffusers kwargs on older torch builds
_orig_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


def _scaled_dot_product_attention_compat(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_scaled_dot_product_attention(*args, **kwargs)


torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention_compat

_orig_torch_linspace = torch.linspace


def _linspace_cast_scalar(x):
    if isinstance(x, torch.Tensor) and x.dtype in (torch.bfloat16, torch.float16):
        return x.to(dtype=torch.float32)
    return x


def _torch_linspace_compat(start, end, steps=100, *args, **kwargs):
    start_cast = _linspace_cast_scalar(start)
    end_cast = _linspace_cast_scalar(end)
    requested_dtype = kwargs.get("dtype")
    needs_downcast = requested_dtype in (torch.bfloat16, torch.float16)
    if needs_downcast:
        kwargs["dtype"] = torch.float32

    result = _orig_torch_linspace(start_cast, end_cast, steps, *args, **kwargs)

    if needs_downcast:
        result = result.to(requested_dtype)

    return result


torch.linspace = _torch_linspace_compat

if hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, 'linspace'):
    _orig_aten_linspace_default = torch.ops.aten.linspace.default
    def _aten_linspace_default_compat(start, end, steps=100, dtype=None, layout=None, device=None, pin_memory=None):
        start_cast = _linspace_cast_scalar(start)
        end_cast = _linspace_cast_scalar(end)
        requested_dtype = dtype
        needs_downcast = requested_dtype in (torch.bfloat16, torch.float16)
        dtype_internal = torch.float32 if needs_downcast else requested_dtype
        result = _orig_aten_linspace_default(start_cast, end_cast, steps, dtype_internal, layout, device, pin_memory)
        if needs_downcast and isinstance(result, torch.Tensor) and result.dtype != requested_dtype:
            result = result.to(requested_dtype)
        return result
    torch.ops.aten.linspace.default = _aten_linspace_default_compat


def _cast_trainable_params_to_fp32(module):
    for param in module.parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.to(torch.float32)


def _cast_trainable_grads_to_fp32(module):
    for param in module.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        target_dtype = torch.float32 if param.dtype == torch.float32 else param.dtype
        if param.grad.dtype != target_dtype:
            param.grad.data = param.grad.data.to(target_dtype)


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

    
    
## SDXL
import functools
import gc
from torchvision.transforms.functional import crop, center_crop
from transformers import AutoTokenizer, PretrainedConfig

from utils.validation_metrics import (
    aggregate_metrics,
    build_metric_record,
    compute_niqe,
    compute_pairwise_metrics,
)
from utils.lut_utils import CubeLUT, LUTPicker, apply_lut, load_cube_luts
from utils.flux_image import choose_flux_resolution, prepare_flux_images



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

DEFAULT_FLUX_LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_add_out",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
)


def flux_tokenize_captions(
    captions,
    proportion_empty_prompts,
    tokenizer_one,
    tokenizer_two,
    max_sequence_length,
    is_train=True,
):
    processed = []
    for caption in captions:
        if random.random() < proportion_empty_prompts:
            processed.append("")
        elif isinstance(caption, str):
            processed.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            processed.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError("Caption entries must be strings or iterables of strings for FLUX prompts.")
    clip_inputs = tokenizer_one(
        processed,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    t5_inputs = tokenizer_two(
        processed,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    return processed, clip_inputs.input_ids, t5_inputs.input_ids


def flux_encode_prompt_batch(
    clip_input_ids,
    t5_input_ids,
    text_encoder_one,
    text_encoder_two,
    device,
    dtype,
):
    clip_ids = clip_input_ids.to(device)
    t5_ids = t5_input_ids.to(device)
    with torch.no_grad():
        clip_outputs = text_encoder_one(clip_ids, output_hidden_states=False)
        pooled = clip_outputs.pooler_output
        t5_outputs = text_encoder_two(t5_ids)
        prompt_embeds = t5_outputs[0]
    pooled = pooled.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    return prompt_embeds, pooled, text_ids


        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="flux",
        choices=("sd15", "sdxl", "flux"),
        help="Model family to fine-tune. Defaults to the FLUX-Kontext architecture.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    # Custom dataset support (pairwise user preferences)
    parser.add_argument(
        "--custom_data_root",
        type=str,
        default=None,
        help=(
            "Root folder for custom dataset with 'images/' and 'responses/' subfolders."
        ),
    )
    parser.add_argument(
        "--user_json_path",
        type=str,
        default=None,
        help=(
            "Path to user response JSON (e.g., datasets/responses/<user_id>.json) for custom pairwise dataset."
        ),
    )
    parser.add_argument(
        "--custom_n_train",
        type=int,
        default=60,
        help="Number of training samples for custom pairwise dataset split (rest go to test).",
    )
    parser.add_argument(
        "--caption_default",
        type=str,
        default="",
        help="Default caption text to use for custom dataset items (empty for unconditional).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    parser.add_argument(
        "--disable_xformers",
        action="store_true",
        help="Disable xFormers attention and use PyTorch SDPA if available (recommended on H100 without xFormers SM90).",
    )
    parser.add_argument(
        "--flux_use_lora",
        action="store_true",
        help="Enable LoRA adapters for the FLUX transformer to reduce GPU memory.",
    )
    parser.add_argument(
        "--flux_lora_r",
        type=int,
        default=16,
        help="Rank used for FLUX LoRA adapters (ignored when --flux_use_lora is not set).",
    )
    parser.add_argument(
        "--flux_lora_alpha",
        type=int,
        default=32,
        help="Alpha scaling factor for FLUX LoRA adapters.",
    )
    parser.add_argument(
        "--flux_lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied inside FLUX LoRA adapters.",
    )
    parser.add_argument(
        "--flux_lora_targets",
        type=str,
        default=None,
        help="Comma separated list of module name substrings to target with FLUX LoRA. Uses attention projections by default.",
    )
    parser.add_argument(
        "--flux_lora_adapter_name",
        type=str,
        default="flux_lora",
        help="Adapter identifier used when registering FLUX LoRA weights.",
    )
    parser.add_argument(
        "--flux_aux_ratio",
        type=float,
        default=0.2,
        help="Probability of sampling auxiliary (preferred, generated) pairs during FLUX DPO training. Set to 0 to disable.",
    )
    parser.add_argument(
        "--flux_aux_prompt_template",
        type=str,
        default=(
            "{caption} High-fidelity structural enhancement, correct colors and lighting while preserving composition,"
            " fine textures, and subject integrity."
        ),
        help=(
            "Prompt used for FLUX-Kontext img2img auxiliary generations. If the template contains '{caption}',"
            " it will be replaced with the sample caption (or removed when unavailable)."
        ),
    )
    parser.add_argument(
        "--flux_aux_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for auxiliary FLUX-Kontext img2img generations.",
    )
    parser.add_argument(
        "--flux_aux_inference_steps",
        type=int,
        default=24,
        help="Number of inference steps for auxiliary FLUX-Kontext img2img generations.",
    )
    parser.add_argument(
        "--flux_aux_force_regenerate",
        action="store_true",
        help="Regenerate cached auxiliary images even if existing files are detected.",
    )
    parser.add_argument(
        "--flux_aux_lut_dir",
        type=str,
        default=None,
        help="Directory containing .cube LUT files. When set, auxiliary images are created by applying a random LUT to the non-preferred image instead of running FLUX img2img.",
    )
    parser.add_argument(
        "--flux_aux_lut_order",
        type=str,
        default="red_fastest",
        choices=("red_fastest", "blue_fastest"),
        help="Data ordering of .cube files when parsing LUTs (use 'red_fastest' for most film LUTs).",
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    # Train logging interval (for scalars)
    parser.add_argument(
        "--log_scalar_steps",
        type=int,
        default=100,
        help="Log scalar metrics every N steps (default 100).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="DPO-PIE-Flux",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # Validation controls (baseline vs trained, img2img)
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="If > 0, run validation (baseline vs trained) every N steps.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of test images to use during each validation event.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=20,
        help="Num inference steps for validation img2img runs.",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for validation img2img runs.",
    )
    parser.add_argument(
        "--validation_strength",
        type=float,
        default=0.6,
        help="Strength parameter for img2img validation (amount of noise to add).",
    )
    parser.add_argument(
        "--validation_prompts",
        nargs='*',
        default=None,
        help="Optional list of prompts for validation; falls back to dataset caption or defaults.",
    )
    parser.add_argument(
        "--validation_use_baseline_cache",
        action="store_true",
        help="Cache baseline validation outputs and reuse them to avoid repeated pretrained inference.",
    )
    parser.add_argument(
        "--validation_baseline_cache_dir",
        type=str,
        default=None,
        help="Directory for stored baseline validation outputs (defaults to <output_dir>/validation_baseline_cache).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Scalar guidance value passed to FLUX transformer when guidance embeddings are enabled.",
    )
    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--beta_dpo_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Strategy used to adjust beta_dpo over optimizer steps.",
    )
    parser.add_argument(
        "--beta_dpo_start",
        type=float,
        default=None,
        help="Initial beta value when scheduling is enabled; defaults to --beta_dpo.",
    )
    parser.add_argument(
        "--beta_dpo_schedule_steps",
        type=int,
        default=0,
        help="Number of optimizer steps used to transition from the start beta to --beta_dpo. Zero keeps beta constant.",
    )
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--dreamlike_pairs_only", action="store_true", help="Only train on pairs where both generations are from dreamlike"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_data_dir is None
        and args.custom_data_root is None
    ):
        raise ValueError("Need either a dataset name, a training folder, or custom_data_root.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    args.train_method = 'sft' if args.sft else 'dpo'
    # Ensure downstream checks expecting a string don't fail when using custom dataset
    if args.custom_data_root and (args.dataset_name is None):
        args.dataset_name = 'custom_pairwise'
    return args


def build_beta_schedule(args):
    target = args.beta_dpo
    start = args.beta_dpo_start if args.beta_dpo_start is not None else target
    schedule_kind = (args.beta_dpo_schedule or "constant").lower()
    steps = args.beta_dpo_schedule_steps

    if schedule_kind == "constant" or steps <= 0 or math.isclose(start, target, rel_tol=1e-8, abs_tol=1e-8):
        return lambda step: target

    if steps == 1:
        return lambda step: target

    denom = steps - 1

    def compute_progress(step: int) -> float:
        clamped = min(max(step, 0), denom)
        return clamped / denom

    if schedule_kind == "linear":
        def schedule(step: int) -> float:
            progress = compute_progress(step)
            return start + (target - start) * progress
    elif schedule_kind == "cosine":
        def schedule(step: int) -> float:
            progress = compute_progress(step)
            weight = 0.5 * (1 - math.cos(math.pi * progress))
            return start + (target - start) * weight
    else:
        return lambda step: target

    return schedule


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    args.model_arch = args.model_arch.lower()
    if getattr(args, "sdxl", False):
        args.model_arch = "sdxl"
    args.sdxl = args.model_arch == "sdxl"
    args.flux = args.model_arch == "flux"
    beta_schedule_fn = build_beta_schedule(args) if args.train_method == 'dpo' else None
    beta_schedule_start = args.beta_dpo_start if args.beta_dpo_start is not None else args.beta_dpo
    beta_schedule_is_dynamic = (
        args.train_method == 'dpo'
        and args.beta_dpo_schedule != "constant"
        and args.beta_dpo_schedule_steps > 0
        and not math.isclose(beta_schedule_start, args.beta_dpo, rel_tol=1e-8, abs_tol=1e-8)
    )

    flux_lora_target_modules = None
    flux_lora_adapter_name = None
    if args.flux and args.flux_use_lora:
        if args.flux_lora_targets:
            flux_lora_target_modules = [module.strip() for module in args.flux_lora_targets.split(",") if module.strip()]
        else:
            flux_lora_target_modules = list(DEFAULT_FLUX_LORA_TARGET_MODULES)
        flux_lora_adapter_name = args.flux_lora_adapter_name or "flux_lora"
        args.resolved_flux_lora_targets = ",".join(flux_lora_target_modules)


    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.train_method == 'dpo' and accelerator.is_main_process:
        if beta_schedule_is_dynamic:
            logger.info(
                "Using %s beta_dpo schedule (start=%.4g, target=%.4g, steps=%d)",
                args.beta_dpo_schedule,
                beta_schedule_start,
                args.beta_dpo,
                args.beta_dpo_schedule_steps,
            )
        else:
            logger.info("Using constant beta_dpo=%.4g", args.beta_dpo)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    if args.flux:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        tokenizer = None
        tokenizer_and_encoder_name = args.pretrained_model_name_or_path
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        noise_scheduler_copy = None

        def enforce_zero_terminal_snr(scheduler):
            # Modified from https://arxiv.org/pdf/2305.08891.pdf
            # Turbo needs zero terminal SNR to truly learn from noise
            # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
            # Convert betas to alphas_bar_sqrt
            alphas = 1 - scheduler.betas
            alphas_bar = alphas.cumprod(0)
            alphas_bar_sqrt = alphas_bar.sqrt()

            # Store old values.
            alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
            alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
            # Shift so last timestep is zero.
            alphas_bar_sqrt -= alphas_bar_sqrt_T
            # Scale so first timestep is back to old value.
            alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

            alphas_bar = alphas_bar_sqrt ** 2
            alphas = alphas_bar[1:] / alphas_bar[:-1]
            alphas = torch.cat([alphas_bar[0:1], alphas])

            alphas_cumprod = torch.cumprod(alphas, dim=0)
            scheduler.alphas_cumprod = alphas_cumprod
            return

        if "turbo" in args.pretrained_model_name_or_path:
            enforce_zero_terminal_snr(noise_scheduler)

        # SDXL has two text encoders
        if args.sdxl:
            # Load the tokenizers
            if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-refiner-1.0":
                tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
            else:
                tokenizer_and_encoder_name = args.pretrained_model_name_or_path
            tokenizer_one = AutoTokenizer.from_pretrained(
                tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False
            )
            tokenizer_two = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
            )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if args.flux:
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            text_encoders = [text_encoder_one, text_encoder_two]
            tokenizers = [tokenizer_one, tokenizer_two]
        elif args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
               tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
        )
        if args.flux:
            ref_unet = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
            )
        else:
            # clone of model
            ref_unet = UNet2DConditionModel.from_pretrained(
                args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                subfolder="unet", revision=args.revision
            )
    if args.unet_init and not args.flux:
        print("Initializing unet from", args.unet_init)
    if args.flux:
        unet = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

    unet_parameters_to_optimize = unet.parameters()
    if args.flux and args.flux_use_lora:
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise ImportError(
                "LoRA support requires the `peft` package. Install it with `pip install peft>=0.6.0` before rerunning."
            ) from exc

        lora_config = LoraConfig(
            r=args.flux_lora_r,
            lora_alpha=args.flux_lora_alpha,
            lora_dropout=args.flux_lora_dropout,
            target_modules=flux_lora_target_modules,
            init_lora_weights="gaussian",
        )
        adapter_name = flux_lora_adapter_name or "flux_lora"
        if not flux_lora_target_modules:
            raise ValueError("No target modules resolved for FLUX LoRA adapters.")
        unet.add_adapter(lora_config, adapter_name=adapter_name)
        unet.set_adapter(adapter_name)
        unet.enable_lora()

        for param in unet.parameters():
            param.requires_grad = False
        for name, param in unet.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        unet_parameters_to_optimize = [param for param in unet.parameters() if param.requires_grad]
        if not unet_parameters_to_optimize:
            raise ValueError("LoRA adapters were enabled but no parameters are marked trainable.")

        _cast_trainable_params_to_fp32(unet)

        trainable_param_count = sum(p.numel() for p in unet_parameters_to_optimize)
        logger.info(
            "Enabled FLUX LoRA adapter '%s' targeting %d modules (rank=%d, alpha=%d, dropout=%.2f). Trainable params: %s",
            adapter_name,
            len(flux_lora_target_modules),
            args.flux_lora_r,
            args.flux_lora_alpha,
            args.flux_lora_dropout,
            f"{trainable_param_count:,}",
        )

    flux_max_sequence_length = (
        getattr(text_encoder_two.config, "max_position_embeddings", 512) if args.flux else None
    )
    vae_shift_factor = getattr(vae.config, "shift_factor", 0.0) if args.flux else None
    vae_scaling_factor = getattr(vae.config, "scaling_factor", 1.0) if args.flux else None
    flux_vae_scale = 2 ** (len(vae.config.block_out_channels) - 1) if args.flux else None

    # Freeze vae, text_encoder(s), reference denoiser
    vae.requires_grad_(False)
    if args.flux:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    elif args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == 'dpo': ref_unet.requires_grad_(False)

    # xformers efficient attention
    if (not args.flux) and is_xformers_available() and (not args.disable_xformers):
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17."
            )
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Failed to enable xFormers attention, falling back to PyTorch SDPA: {e}")
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    unet.set_attn_processor(AttnProcessor2_0())
                    logger.info("Using PyTorch SDPA attention (AttnProcessor2_0)")
            except Exception as ee:
                logger.warning(f"Could not set SDPA attention: {ee}")
    elif not args.flux:
        # xformers disabled or not available; try SDPA, otherwise use default attention
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                unet.set_attn_processor(AttnProcessor2_0())
                logger.info("xFormers disabled/not available; using PyTorch SDPA attention")
        except Exception as e:
            logger.info(f"xFormers disabled/not available; using default attention. Reason: {e}")
    else:
        logger.info("Skipping attention backend adjustments for Flux transformer.")

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            
            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            save_subfolder = "transformer" if args.flux else "unet"
            for i, model in enumerate(models_to_save):
                if args.flux and args.flux_use_lora:
                    adapter_dir = os.path.join(output_dir, f"{save_subfolder}_lora")
                    model.save_lora_adapter(adapter_dir, adapter_name=flux_lora_adapter_name or "flux_lora")
                else:
                    model.save_pretrained(os.path.join(output_dir, save_subfolder))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            save_subfolder = "transformer" if args.flux else "unet"
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                if args.flux and args.flux_use_lora:
                    adapter_dir = os.path.join(input_dir, f"{save_subfolder}_lora")
                    if os.path.isdir(adapter_dir):
                        model.load_lora_adapter(adapter_dir, adapter_name=flux_lora_adapter_name or "flux_lora")
                        model.set_adapter(flux_lora_adapter_name or "flux_lora")
                        model.enable_lora()
                else:
                    # load diffusers style into model
                    if args.flux:
                        load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                    else:
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl or args.flux: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL/FLUX")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(unet_parameters_to_optimize,
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            unet_parameters_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        
        
        
    # Build dataset
    # - Prefer explicit custom dataset if provided
    # - Else fall back to HF hub dataset or generic imagefolder
    if args.custom_data_root and args.user_json_path:
        from utils.custom_dataset import build_user_pairwise_dataset

        dataset = build_user_pairwise_dataset(
            data_root=args.custom_data_root,
            user_json_path=args.user_json_path,
            n_train=args.custom_n_train,
            seed=args.seed,
            caption_default=args.caption_default,
        )
    else:
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files[args.split] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset[args.split].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if 'pickapic' in args.dataset_name or (args.train_method == 'dpo'):
        pass
    elif args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def _format_aux_prompt(raw_caption: str | None) -> str:
        template = args.flux_aux_prompt_template or ""
        caption_text = (raw_caption or "").strip()
        if "{caption}" in template:
            return template.replace("{caption}", caption_text)
        if caption_text and template:
            return f"{caption_text}. {template}"
        if caption_text:
            return caption_text
        return template

    def _prepare_flux_auxiliary_split(split_name: str, split_ds) :
        if args.flux_aux_force_regenerate and "jpg_generated" in split_ds.column_names:
            split_ds = split_ds.remove_columns(["jpg_generated"])
        if "jpg_generated" in split_ds.column_names:
            return split_ds

        cache_root = Path(args.output_dir) / "flux_aux_cache"
        cache_dir = cache_root / split_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        num_rows = split_ds.num_rows
        payloads: List[bytes | None] = [None] * num_rows
        missing_indices: List[int] = []

        for idx in range(num_rows):
            cache_path = cache_dir / f"{idx:06d}.png"
            if args.flux_aux_force_regenerate and cache_path.exists():
                cache_path.unlink()
            if cache_path.exists():
                payloads[idx] = cache_path.read_bytes()
            else:
                missing_indices.append(idx)

        pipe = None
        if accelerator.is_main_process and args.flux and args.train_method == 'dpo' and args.flux_aux_ratio > 0.0:
            if missing_indices:
                logger.info(
                    "Generating %d auxiliary FLUX-Kontext samples for split '%s'",
                    len(missing_indices),
                    split_name,
                )
                aux_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16
                pipe = FluxKontextPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    torch_dtype=aux_dtype,
                ).to(accelerator.device, torch_dtype=aux_dtype)
                pipe.set_progress_bar_config(disable=True)
                if hasattr(pipe, "safety_checker"):
                    pipe.safety_checker = None

                for local_idx, idx in enumerate(missing_indices):
                    example = split_ds[idx]
                    label = example.get("label_0", 1)
                    img0 = Image.open(io.BytesIO(example["jpg_0"])).convert("RGB")
                    img1 = Image.open(io.BytesIO(example["jpg_1"])).convert("RGB")
                    if label == 1:
                        preferred_raw = img0
                        non_preferred_raw = img1
                    else:
                        preferred_raw = img1
                        non_preferred_raw = img0

                    target_size = choose_flux_resolution([preferred_raw.size, non_preferred_raw.size])
                    (nonpref_for_pipe,) = prepare_flux_images(
                        (non_preferred_raw,),
                        target_size,
                        crop_u=0.5,
                        crop_v=0.5,
                        flip=False,
                    )
                    prompt = _format_aux_prompt(example.get("caption"))

                    generator = torch.Generator(device=accelerator.device)
                    base_seed = args.seed or 0
                    generator.manual_seed(int(base_seed + idx))

                    with torch.no_grad():
                        output = pipe(
                            image=nonpref_for_pipe,
                            prompt=prompt,
                            guidance_scale=args.flux_aux_guidance_scale,
                            num_inference_steps=args.flux_aux_inference_steps,
                            generator=generator,
                        )

                    candidate = output.images[0].convert("RGB")
                    (processed_candidate,) = prepare_flux_images(
                        (candidate,),
                        target_size,
                        crop_u=0.5,
                        crop_v=0.5,
                        flip=False,
                    )

                    cache_path = cache_dir / f"{idx:06d}.png"
                    processed_candidate.save(cache_path, format="PNG")
                    payloads[idx] = cache_path.read_bytes()

        for idx in range(num_rows):
            if payloads[idx] is None:
                cache_path = cache_dir / f"{idx:06d}.png"
                if not cache_path.exists():
                    raise FileNotFoundError(
                        f"Expected cached auxiliary image at {cache_path} for split '{split_name}' index {idx}."
                    )
                payloads[idx] = cache_path.read_bytes()

        if pipe is not None:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        split_ds = split_ds.add_column("jpg_generated", payloads)  # type: ignore[arg-type]
        return split_ds

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    
    ##### START BIG OLD DATASET BLOCK #####
    
    #### START PREPROCESSING/COLLATION ####
    if args.train_method == 'dpo':
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")
        def preprocess_train(examples):
            if args.flux:
                combined_pixel_values = []
                aux_pixel_values = []
                has_generated = "jpg_generated" in examples
                lut_enabled = bool(available_luts) and args.flux_aux_ratio > 0.0

                if lut_enabled and not hasattr(preprocess_train, "_lut_picker"):
                    seed_base = args.seed if args.seed is not None else 0
                    picker_seed = seed_base + accelerator.process_index * 7919
                    preprocess_train._lut_picker = LUTPicker(available_luts, seed=picker_seed)
                lut_picker_local = getattr(preprocess_train, "_lut_picker", None)

                for idx, label_0 in enumerate(examples["label_0"]):
                    img0 = Image.open(io.BytesIO(examples["jpg_0"][idx])).convert("RGB")
                    img1 = Image.open(io.BytesIO(examples["jpg_1"][idx])).convert("RGB")

                    if label_0 == 1:
                        preferred_raw = img0
                        non_preferred_raw = img1
                    else:
                        preferred_raw = img1
                        non_preferred_raw = img0

                    generated_raw = None
                    if has_generated:
                        gen_bytes = examples["jpg_generated"][idx]
                        if isinstance(gen_bytes, (bytes, bytearray)):
                            try:
                                generated_raw = Image.open(io.BytesIO(gen_bytes)).convert("RGB")
                            except Exception:
                                generated_raw = None

                    generated_source = (
                        non_preferred_raw if (generated_raw is None and lut_enabled) else generated_raw
                    )

                    size_candidates = [preferred_raw.size, non_preferred_raw.size]
                    if generated_source is not None:
                        size_candidates.append(generated_source.size)

                    target_size = choose_flux_resolution(size_candidates)
                    if args.random_crop:
                        crop_u = random.random()
                        crop_v = random.random()
                    else:
                        crop_u = crop_v = 0.5
                    flip_pair = (not args.no_hflip) and (random.random() < 0.5)

                    proc_pref, proc_nonpref, proc_gen = prepare_flux_images(
                        (preferred_raw, non_preferred_raw, generated_source),
                        target_size,
                        crop_u=crop_u,
                        crop_v=crop_v,
                        flip=flip_pair,
                    )

                    def _tensorize(image: Image.Image) -> torch.Tensor:
                        tensor = TF.to_tensor(image)
                        tensor = TF.normalize(tensor, [0.5], [0.5])
                        return tensor

                    pref_tensor = _tensorize(proc_pref)
                    nonpref_tensor = _tensorize(proc_nonpref)
                    pair_tensor = (pref_tensor, nonpref_tensor)

                    if args.choice_model:
                        if label_0 == 1:
                            combined = torch.cat(pair_tensor, dim=0)
                        else:
                            combined = torch.cat(pair_tensor[::-1], dim=0)
                    else:
                        combined = torch.cat(pair_tensor, dim=0)
                    combined_pixel_values.append(combined)

                    proc_gen_image = None
                    if lut_enabled:
                        if lut_picker_local is None:
                            raise RuntimeError("LUTPicker not initialised for LUT-based auxiliary mode.")
                        selected_lut = lut_picker_local.pick()
                        proc_gen_image = apply_lut(proc_nonpref, selected_lut)
                    elif proc_gen is not None:
                        proc_gen_image = proc_gen

                    if proc_gen_image is not None:
                        gen_tensor = _tensorize(proc_gen_image)
                        aux_pair = torch.cat((pref_tensor, gen_tensor), dim=0)
                    else:
                        aux_pair = combined
                    aux_pixel_values.append(aux_pair)

                examples["pixel_values"] = combined_pixel_values
                examples["pixel_values_generated"] = aux_pixel_values
            else:
                all_pixel_values = []
                for col_name in ['jpg_0', 'jpg_1']:
                    images = [Image.open(io.BytesIO(im_bytes)).convert("RGB")
                                for im_bytes in examples[col_name]]
                    pixel_values = [train_transforms(image) for image in images]
                    all_pixel_values.append(pixel_values)
                im_tup_iterator = zip(*all_pixel_values)
                combined_pixel_values = []
                for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
                    if label_0==0 and (not args.choice_model):
                        im_tup = im_tup[::-1]
                    if len(im_tup) > 1:
                        min_h = min(t.shape[-2] for t in im_tup)
                        min_w = min(t.shape[-1] for t in im_tup)
                        if any((t.shape[-2] != min_h) or (t.shape[-1] != min_w) for t in im_tup):
                            im_tup = tuple(center_crop(t, [min_h, min_w]) for t in im_tup)
                    combined_im = torch.cat(im_tup, dim=0)
                    combined_pixel_values.append(combined_im)
                examples["pixel_values"] = combined_pixel_values
            # SDXL takes raw prompts
            if args.flux:
                captions, clip_ids, t5_ids = flux_tokenize_captions(
                    list(examples[caption_column]),
                    args.proportion_empty_prompts,
                    tokenizer_one,
                    tokenizer_two,
                    flux_max_sequence_length,
                    is_train=True,
                )
                examples["caption"] = captions
                examples["clip_input_ids"] = [ids for ids in clip_ids]
                examples["t5_input_ids"] = [ids for ids in t5_ids]
            elif not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            if args.flux:
                aux_values = torch.stack([example["pixel_values_generated"] for example in examples])
                aux_values = aux_values.to(memory_format=torch.contiguous_format).float()
                return_d["pixel_values_generated"] = aux_values
            # SDXL takes raw prompts
            if args.flux:
                return_d["clip_input_ids"] = torch.stack([example["clip_input_ids"] for example in examples])
                return_d["t5_input_ids"] = torch.stack([example["t5_input_ids"] for example in examples])
                return_d["caption"] = [example["caption"] for example in examples]
            elif args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
                
            if args.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ['jpg_0', 'jpg_1']:
                    return_d[k] = [Image.open(io.BytesIO( example[k])).convert("RGB")
                                   for example in examples]
                return_d["caption"] = [example["caption"] for example in examples] 
            return return_d
         
        if args.choice_model:
            # TODO: Fancy way of doing this?
            if args.choice_model == 'hps':
                from utils.hps_utils import Selector
            elif args.choice_model == 'clip':
                from utils.clip_utils import Selector
            elif args.choice_model == 'pickscore':
                from utils.pickscore_utils import Selector
            elif args.choice_model == 'aes':
                from utils.aes_utils import Selector
            selector = Selector('cpu' if args.sdxl else accelerator.device)

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]
            def choice_model_says_flip(batch):
                assert len(batch['caption'])==1 # Can switch to iteration but not needed for nwo
                return do_flip(batch['jpg_0'][0], batch['jpg_1'][0], batch['caption'][0])
    elif args.train_method == 'sft':
        def preprocess_train(examples):
            if 'pickapic' in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0']):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0==1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if args.flux:
                captions, clip_ids, t5_ids = flux_tokenize_captions(
                    list(examples[caption_column]),
                    args.proportion_empty_prompts,
                    tokenizer_one,
                    tokenizer_two,
                    flux_max_sequence_length,
                    is_train=True,
                )
                examples["caption"] = captions
                examples["clip_input_ids"] = [ids for ids in clip_ids]
                examples["t5_input_ids"] = [ids for ids in t5_ids]
            elif not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            if args.flux:
                return_d["clip_input_ids"] = torch.stack([example["clip_input_ids"] for example in examples])
                return_d["t5_input_ids"] = torch.stack([example["t5_input_ids"] for example in examples])
                return_d["caption"] = [example["caption"] for example in examples]
            elif args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d
    #### END PREPROCESSING/COLLATION ####
    
    available_luts = None
    if (
        args.flux
        and args.train_method == 'dpo'
        and args.flux_aux_ratio > 0.0
        and args.flux_aux_lut_dir
    ):
        lut_dir = Path(args.flux_aux_lut_dir)
        if not lut_dir.exists():
            raise ValueError(f"LUT directory {lut_dir} does not exist")
        available_luts = load_cube_luts(lut_dir, order_hint=args.flux_aux_lut_order)
        logger.info(
            "Loaded %d LUTs from %s for auxiliary training", len(available_luts), lut_dir
        )

    ### DATASET #####
    with accelerator.main_process_first():
        if (
            args.flux
            and args.train_method == 'dpo'
            and args.flux_aux_ratio > 0.0
            and not available_luts
        ):
            candidate_splits = {args.split}
            if "test" in dataset:
                candidate_splits.add("test")
            for split_name in sorted(candidate_splits):
                if split_name in dataset:
                    dataset[split_name] = _prepare_flux_auxiliary_split(split_name, dataset[split_name])
        if 'pickapic' in args.dataset_name:
            # eliminate no-decisions (0.5-0.5 labels)
            orig_len = dataset[args.split].num_rows
            not_split_idx = [i for i,label_0 in enumerate(dataset[args.split]['label_0'])
                             if label_0 in (0,1) ]
            dataset[args.split] = dataset[args.split].select(not_split_idx)
            new_len = dataset[args.split].num_rows
            print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic")
            
            # Below if if want to train on just the Dreamlike vs dreamlike pairs
            if args.dreamlike_pairs_only:
                orig_len = dataset[args.split].num_rows
                dream_like_idx = [i for i,(m0,m1) in enumerate(zip(dataset[args.split]['model_0'],
                                                                   dataset[args.split]['model_1']))
                                  if ( ('dream' in m0) and ('dream' in m1) )]
                dataset[args.split] = dataset[args.split].select(dream_like_idx)
                new_len = dataset[args.split].num_rows
                print(f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic")
                
        if args.max_train_samples is not None:
            dataset[args.split] = dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset[args.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split=='train'),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    unet_trainable_fp32 = accelerator.unwrap_model(unet)
    _cast_trainable_params_to_fp32(unet_trainable_fp32)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    if args.flux:
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    elif args.sdxl:
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        print("Offloading text encoders to cpu")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            print("offload ref_unet")
            ref_unet = accelerate.cpu_offload(ref_unet)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Determine if wandb is active for this run
    def _is_wandb_active():
        try:
            log_with = accelerator.log_with
            loggers = log_with if isinstance(log_with, (list, tuple, set)) else [log_with]
            for logger_entry in loggers:
                name = getattr(logger_entry, "value", logger_entry)
                if name in ("wandb", "all"):
                    return is_wandb_available()
            return False
        except Exception:
            return False

    if args.flux:
        def get_flux_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = []
            for t in timesteps:
                matches = (schedule_timesteps == t).nonzero(as_tuple=True)[0]
                if matches.numel() == 0:
                    # `FlowMatchEulerDiscreteScheduler` does not explicitly include timestep 0 in its
                    # schedule (only appending an extra sigma=0 entry). Map missing entries to the
                    # closest available timestep so training can proceed without crashing.
                    if t.item() == 0:
                        step_indices.append(sigmas.shape[0] - 1)
                        continue
                    closest = torch.argmin(torch.abs(schedule_timesteps - t))
                    step_indices.append(closest.item())
                else:
                    step_indices.append(matches[0].item())

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

    # Validation: compare baseline vs trained (img2img) on held-out test images
    _val_pipe_trained = None
    _val_pipe_base = None
    _val_vae_eval = None
    _val_text_encoder_eval = None
    _val_text_encoder_two_eval = None

    baseline_cache_dir = (
        Path(args.validation_baseline_cache_dir)
        if args.validation_baseline_cache_dir
        else Path(args.output_dir) / "validation_baseline_cache"
    )
    _baseline_cache_enabled = bool(args.validation_use_baseline_cache)
    _baseline_cache_ready = False
    _baseline_cache_samples: Dict[int, Dict[str, Dict[str, Any]]] = {}

    def _ensure_validation_pipelines(require_base: bool = True):
        nonlocal _val_pipe_trained, _val_pipe_base
        nonlocal _val_vae_eval, _val_text_encoder_eval, _val_text_encoder_two_eval
        if not accelerator.is_main_process:
            return None, None
        base_needed = require_base or (not _baseline_cache_enabled) or (not _baseline_cache_ready)
        if args.flux:
            validation_dtype = weight_dtype
            if _val_pipe_trained is None:
                if _val_vae_eval is None:
                    _val_vae_eval = vae
                    _val_vae_eval.to(accelerator.device, dtype=validation_dtype)
                    _val_vae_eval.eval()
                if _val_text_encoder_eval is None:
                    _val_text_encoder_eval = text_encoder_one
                    _val_text_encoder_eval.to(accelerator.device, dtype=validation_dtype)
                    _val_text_encoder_eval.eval()
                if _val_text_encoder_two_eval is None:
                    _val_text_encoder_two_eval = text_encoder_two
                    _val_text_encoder_two_eval.to(accelerator.device, dtype=validation_dtype)
                    _val_text_encoder_two_eval.eval()

                transformer_eval = accelerator.unwrap_model(unet)
                transformer_eval.to(accelerator.device, dtype=validation_dtype)
                transformer_eval.eval()
                _val_pipe_trained = FluxKontextPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    transformer=transformer_eval,
                    vae=_val_vae_eval,
                    text_encoder=_val_text_encoder_eval,
                    text_encoder_2=_val_text_encoder_two_eval,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    scheduler=copy.deepcopy(noise_scheduler_copy),
                    torch_dtype=validation_dtype,
                )
                _val_pipe_trained = _val_pipe_trained.to(accelerator.device, torch_dtype=validation_dtype)
                _val_pipe_trained.set_progress_bar_config(disable=True)
            if base_needed:
                if _val_pipe_base is None:
                    transformer_base_eval = ref_unet
                    transformer_base_eval.to(accelerator.device, dtype=validation_dtype)
                    transformer_base_eval.eval()
                    _val_pipe_base = FluxKontextPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        revision=args.revision,
                        transformer=transformer_base_eval,
                        vae=_val_vae_eval,
                        text_encoder=_val_text_encoder_eval,
                        text_encoder_2=_val_text_encoder_two_eval,
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        scheduler=copy.deepcopy(noise_scheduler_copy),
                        torch_dtype=validation_dtype,
                    )
                    _val_pipe_base = _val_pipe_base.to(accelerator.device, torch_dtype=validation_dtype)
                    _val_pipe_base.set_progress_bar_config(disable=True)
            elif _val_pipe_base is not None:
                _val_pipe_base = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif args.sdxl:
            if _val_pipe_trained is None:
                _val_pipe_trained = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32
                ).to(accelerator.device, torch_dtype=torch.float32)
                _val_pipe_trained.safety_checker = None
                _val_pipe_trained.set_progress_bar_config(disable=True)
            if base_needed:
                if _val_pipe_base is None:
                    _val_pipe_base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32
                    ).to(accelerator.device, torch_dtype=torch.float32)
                    _val_pipe_base.safety_checker = None
                    _val_pipe_base.set_progress_bar_config(disable=True)
            elif _val_pipe_base is not None:
                _val_pipe_base = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if _val_pipe_trained is None:
                _val_pipe_trained = StableDiffusionImg2ImgPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32
                ).to(accelerator.device, torch_dtype=torch.float32)
                _val_pipe_trained.safety_checker = None
                _val_pipe_trained.set_progress_bar_config(disable=True)
            if base_needed:
                if _val_pipe_base is None:
                    _val_pipe_base = StableDiffusionImg2ImgPipeline.from_pretrained(
                        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=torch.float32
                    ).to(accelerator.device, torch_dtype=torch.float32)
                    _val_pipe_base.safety_checker = None
                    _val_pipe_base.set_progress_bar_config(disable=True)
            elif _val_pipe_base is not None:
                _val_pipe_base = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # keep base as the original weights, and copy trained weights into the trained pipe
        if not args.flux:
            _val_pipe_trained.unet.load_state_dict(accelerator.unwrap_model(unet).state_dict())
            if hasattr(_val_pipe_trained, "vae") and _val_pipe_trained.vae is not None:
                _val_pipe_trained.vae.to(accelerator.device, dtype=torch.float32)
        return _val_pipe_trained, _val_pipe_base

    def _validation_seed(dataset_idx: int, source_name: str) -> int:
        base_seed = args.seed if args.seed is not None else 123
        offsets = {"preferred": 0, "nonpreferred": 1, "generated": 2}
        offset = offsets.get(source_name, 0)
        return base_seed + dataset_idx * 100 + offset

    def _load_baseline_cache_manifest() -> None:
        nonlocal _baseline_cache_ready, _baseline_cache_samples
        manifest_path = baseline_cache_dir / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load validation baseline cache manifest %s: %s", manifest_path, exc)
            return
        if int(manifest.get("version", 0)) != 1:
            logger.warning(
                "Unsupported validation baseline cache manifest version %s; ignoring cache.",
                manifest.get("version"),
            )
            return
        samples_section = manifest.get("samples", {})
        loaded: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for idx_str, sample_entry in samples_section.items():
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(sample_entry, dict):
                continue
            source_map: Dict[str, Dict[str, Any]] = {}
            for source_name, source_info in sample_entry.items():
                if not isinstance(source_info, dict):
                    continue
                file_rel = source_info.get("file")
                if not file_rel:
                    continue
                try:
                    seed_val = int(source_info.get("seed", _validation_seed(idx, source_name)))
                except (TypeError, ValueError):
                    seed_val = _validation_seed(idx, source_name)
                source_map[source_name] = {
                    "file": Path(file_rel),
                    "seed": seed_val,
                }
            if source_map:
                loaded[idx] = source_map
        if loaded:
            _baseline_cache_samples = loaded
            _baseline_cache_ready = True

    def _write_baseline_cache_manifest(sample_map: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
        manifest_path = baseline_cache_dir / "manifest.json"
        serialisable = {
            str(idx): {
                source_name: {
                    "file": (source_info.get("file").as_posix()
                              if isinstance(source_info.get("file"), Path)
                              else str(source_info.get("file"))),
                    "seed": int(source_info.get("seed", _validation_seed(idx, source_name))),
                }
                for source_name, source_info in sample_entry.items()
                if source_info.get("file")
            }
            for idx, sample_entry in sample_map.items()
        }
        payload = {
            "version": 1,
            "seed_scheme": "dataset_idx_linear_v1",
            "seed_base": args.seed if args.seed is not None else 123,
            "samples": serialisable,
        }
        try:
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to write validation baseline cache manifest %s: %s", manifest_path, exc)

    def _validation_seeds_for_index(dataset_idx: int) -> Dict[str, int]:
        if _baseline_cache_ready:
            sample_entry = _baseline_cache_samples.get(dataset_idx)
            if sample_entry:
                return {
                    source_name: int(source_info.get("seed", _validation_seed(dataset_idx, source_name)))
                    for source_name, source_info in sample_entry.items()
                }
        return {
            name: _validation_seed(dataset_idx, name)
            for name in ("preferred", "nonpreferred", "generated")
        }

    def _load_cached_baseline_outputs(dataset_idx: int) -> Dict[str, Image.Image]:
        if not _baseline_cache_ready:
            return {}
        sample_entry = _baseline_cache_samples.get(dataset_idx)
        if not sample_entry:
            return {}
        cached_outputs: Dict[str, Image.Image] = {}
        for source_name, source_info in sample_entry.items():
            path_info = source_info.get("file")
            if isinstance(path_info, Path):
                image_path = baseline_cache_dir / path_info
            else:
                image_path = baseline_cache_dir / str(path_info)
            if not image_path.exists():
                continue
            try:
                with Image.open(image_path) as handle:
                    cached_outputs[source_name] = handle.convert("RGB")
            except Exception as exc:
                logger.debug(
                    "Failed to load cached baseline image for sample %s (%s): %s",
                    dataset_idx,
                    source_name,
                    exc,
                )
        return cached_outputs

    def _ensure_baseline_cache_populated() -> None:
        nonlocal _baseline_cache_ready, _baseline_cache_samples, _val_pipe_base
        if not _baseline_cache_enabled or not accelerator.is_main_process:
            return
        try:
            baseline_cache_dir.mkdir(parents=True, exist_ok=True)
            if not _baseline_cache_ready:
                _load_baseline_cache_manifest()
            if _baseline_cache_ready:
                return
            _, pipe_base = _ensure_validation_pipelines(require_base=True)
            if pipe_base is None:
                logger.warning("Baseline cache requested but baseline pipeline could not be constructed.")
                return
            num_pairs = min(args.num_validation_images, _dataset_test_len())
            if num_pairs <= 0:
                logger.info("Baseline cache requested but no validation samples found; skipping cache population.")
                _baseline_cache_ready = True
                _baseline_cache_samples = {}
                return

            logger.info("Populating baseline validation cache for %s samples", num_pairs)
            sample_map: Dict[int, Dict[str, Dict[str, Any]]] = {}
            for dataset_idx in range(num_pairs):
                example = _get_test_example(dataset_idx)
                if not example:
                    continue

                preferred_orig = example["preferred"]
                non_preferred_orig = example["non_preferred"]
                dataset_generated = example.get("generated")
                caption = example.get("caption", "")

                if args.flux:
                    size_candidates = [preferred_orig.size, non_preferred_orig.size]
                    if dataset_generated is not None:
                        size_candidates.append(dataset_generated.size)
                    target_size = choose_flux_resolution(size_candidates)
                    proc_pref, proc_nonpref, proc_generated = prepare_flux_images(
                        (preferred_orig, non_preferred_orig, dataset_generated),
                        target_size,
                        crop_u=0.5,
                        crop_v=0.5,
                        flip=False,
                    )
                    if proc_generated is None:
                        lut = _get_validation_lut(dataset_idx)
                        if lut is not None and proc_nonpref is not None:
                            proc_generated = apply_lut(proc_nonpref, lut)
                    inputs_for_pipe = {
                        "preferred": proc_pref,
                        "nonpreferred": proc_nonpref,
                        "generated": proc_generated,
                    }
                    size_lookup = {
                        name: (img.width, img.height) if img is not None else None
                        for name, img in inputs_for_pipe.items()
                    }
                else:
                    inputs_for_pipe = {
                        "preferred": preferred_orig,
                        "nonpreferred": non_preferred_orig,
                        "generated": dataset_generated,
                    }
                    size_lookup = {
                        name: (img.width, img.height) if img is not None else None
                        for name, img in inputs_for_pipe.items()
                    }
                if args.flux:
                    prompt = ""
                elif args.validation_prompts and len(args.validation_prompts) > 0:
                    prompt = args.validation_prompts[min(dataset_idx, len(args.validation_prompts) - 1)]
                else:
                    prompt = caption if (isinstance(caption, str) and caption) else ""

                eval_order = ["preferred", "nonpreferred", "generated"]
                source_sequence = [
                    (name, inputs_for_pipe[name]) for name in eval_order if inputs_for_pipe.get(name) is not None
                ]
                if not source_sequence:
                    continue

                seeds_for_sample = _validation_seeds_for_index(dataset_idx)
                sample_entry: Dict[str, Dict[str, Any]] = {}
                for source_name, source_image in source_sequence:
                    generator_seed = seeds_for_sample.get(source_name, _validation_seed(dataset_idx, source_name))
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(generator_seed)

                    if args.flux:
                        size_tuple = size_lookup.get(source_name)
                        if size_tuple is None:
                            continue
                        image_for_pipe = _prepare_pipeline_image(pipe_base, source_image, size_tuple)
                        kwargs = dict(
                            image=image_for_pipe,
                            prompt="",
                            guidance_scale=args.validation_guidance_scale,
                            num_inference_steps=args.validation_inference_steps,
                            generator=generator,
                            height=size_tuple[1],
                            width=size_tuple[0],
                        )
                    else:
                        kwargs = dict(
                            image=source_image,
                            prompt=prompt,
                            strength=args.validation_strength,
                            guidance_scale=args.validation_guidance_scale,
                            num_inference_steps=args.validation_inference_steps,
                            generator=generator,
                        )

                    try:
                        with torch.no_grad():
                            out = pipe_base(**kwargs)
                    except Exception as exc:
                        logger.warning(
                            "Baseline cache generation failed for sample %s (%s): %s",
                            dataset_idx,
                            source_name,
                            exc,
                        )
                        continue

                    output_image = out.images[0] if hasattr(out, "images") and out.images else None
                    if output_image is None:
                        continue

                    rel_dir = Path(f"{dataset_idx:05d}")
                    cache_dir = baseline_cache_dir / rel_dir
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    rel_path = rel_dir / f"{source_name}.png"
                    try:
                        output_image.save(cache_dir / f"{source_name}.png")
                    except Exception as exc:
                        logger.warning(
                            "Failed to save baseline cache image for sample %s (%s): %s",
                            dataset_idx,
                            source_name,
                            exc,
                        )
                        continue

                    sample_entry[source_name] = {"file": rel_path, "seed": generator_seed}

                if sample_entry:
                    sample_map[dataset_idx] = sample_entry

            if sample_map:
                _write_baseline_cache_manifest(sample_map)
                _baseline_cache_samples = sample_map
                _baseline_cache_ready = True
                logger.info("Baseline validation cache populated with %s samples", len(sample_map))
            else:
                logger.warning("Baseline cache population produced no samples; continuing without cache.")
            _val_pipe_base = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            _cast_trainable_params_to_fp32(unet_trainable_fp32)

    def _dataset_test_len():
        try:
            return dataset["test"].num_rows
        except Exception:
            return 0

    def _get_test_example(i):
        ex = dataset["test"][i]
        caption = ex.get("caption", "")
        preferred = None
        non_preferred = None
        generated = None

        if "jpg_0" in ex and "jpg_1" in ex:
            preferred_raw = Image.open(io.BytesIO(ex["jpg_0"])).convert("RGB")
            non_preferred_raw = Image.open(io.BytesIO(ex["jpg_1"])).convert("RGB")
            label_0 = ex.get("label_0", 1)
            if label_0 == 1:
                preferred = preferred_raw
                non_preferred = non_preferred_raw
            else:
                preferred = non_preferred_raw
                non_preferred = preferred_raw
        elif "image" in ex:
            # Single image without preference pair; cannot evaluate pairwise metrics
            preferred = ex["image"].convert("RGB")
        elif "jpg_0" in ex:
            preferred = Image.open(io.BytesIO(ex["jpg_0"])).convert("RGB")

        if "jpg_generated" in ex and isinstance(ex["jpg_generated"], (bytes, bytearray)):
            try:
                generated = Image.open(io.BytesIO(ex["jpg_generated"])).convert("RGB")
            except Exception:
                generated = None

        if preferred is None or non_preferred is None:
            return None

        return {
            "preferred": preferred,
            "non_preferred": non_preferred,
            "generated": generated,
            "caption": caption,
        }

    _validation_luts_cache: Optional[List[CubeLUT]] = None

    def _get_validation_lut(index: int) -> Optional[CubeLUT]:
        nonlocal _validation_luts_cache
        lut_list = available_luts
        if lut_list is None:
            lut_dir = Path(args.flux_aux_lut_dir) if args.flux_aux_lut_dir else None
            if lut_dir is None:
                return None
            if _validation_luts_cache is None:
                if not lut_dir.exists():
                    logger.warning(
                        "Validation LUT directory %s does not exist; generated-input validation will skip LUTs.",
                        lut_dir,
                    )
                    _validation_luts_cache = []
                else:
                    try:
                        _validation_luts_cache = load_cube_luts(
                            lut_dir, order_hint=args.flux_aux_lut_order
                        )
                        logger.info(
                            "Loaded %d LUTs from %s for validation generated inputs",
                            len(_validation_luts_cache),
                            lut_dir,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load LUTs for validation from %s: %s",
                            lut_dir,
                            exc,
                        )
                        _validation_luts_cache = []
            lut_list = _validation_luts_cache
        if not lut_list:
            return None
        seed_base = args.seed if args.seed is not None else 0
        lut_index = (index + seed_base) % len(lut_list)
        return lut_list[lut_index]

    def _prepare_pipeline_image(pipe, image: Image.Image, size_tuple: Optional[Tuple[int, int]]):
        if pipe is None or image is None:
            return image
        processed = pipe.image_processor.preprocess(
            image,
            height=size_tuple[1] if size_tuple is not None else None,
            width=size_tuple[0] if size_tuple is not None else None,
        )
        target_dtype = getattr(pipe.transformer, "dtype", None)
        if target_dtype is None:
            target_dtype = getattr(pipe.vae, "dtype", None)
        if target_dtype is None:
            target_dtype = torch.float32
        return processed.to(device=accelerator.device, dtype=target_dtype)

    def _build_flux_latent_dtype_callback(pipe):
        if not args.flux or pipe is None:
            return None

        def _callback(pipeline, _step_idx, _timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            target_dtype = getattr(pipeline.transformer, "dtype", None)
            if latents is not None and target_dtype is not None and latents.dtype != target_dtype:
                # FlowMatch scheduler can upcast latents to float32; force them back to the model dtype
                callback_kwargs["latents"] = latents.to(dtype=target_dtype)
            return callback_kwargs

        return _callback

    def _run_validation_if_needed(step: int):
        if args.validation_steps <= 0:
            return
        if step == 0 or (step % args.validation_steps != 0):
            return
        if not accelerator.is_main_process:
            return
        if not _is_wandb_active():
            logger.debug("Skipping validation wandb logging; accelerator.log_with=%s", accelerator.log_with)
            return
        if _dataset_test_len() == 0:
            return

        _ensure_baseline_cache_populated()
        require_base = not (_baseline_cache_enabled and _baseline_cache_ready)
        pipe_trained, pipe_base = _ensure_validation_pipelines(require_base=require_base)
        if pipe_trained is None:
            return
        if require_base and pipe_base is None:
            return

        modules_toggled = []
        seen_modules = set()

        def _ensure_eval(module):
            if module is None:
                return
            if module in seen_modules:
                return
            seen_modules.add(module)
            modules_toggled.append((module, module.training))
            module.eval()

        _ensure_eval(getattr(pipe_trained, "unet", None))
        _ensure_eval(getattr(pipe_trained, "transformer", None))
        if pipe_base is not None:
            _ensure_eval(getattr(pipe_base, "unet", None))
            _ensure_eval(getattr(pipe_base, "transformer", None))

        try:
            base_seed = args.seed if args.seed is not None else 123
            n = min(args.num_validation_images, _dataset_test_len())
            indices = list(range(n))

            metrics_groups = defaultdict(list)
            table_rows = []
            samples_for_logging = []

            logger.info(
                "Running validation at step %s over %s preference pairs", step, len(indices)
            )

            if args.flux and args.validation_prompts:
                logger.warning(
                    "Flux Kontext validation ignores `--validation_prompts`; running with empty prompts."
                )

            source_seed_offsets = {"preferred": 0, "nonpreferred": 1, "generated": 2}
            default_prompt = ""

            for local_idx, dataset_idx in enumerate(indices):
                example = _get_test_example(dataset_idx)
                if not example:
                    continue

                preferred_orig = example["preferred"]
                non_preferred_orig = example["non_preferred"]
                dataset_generated = example.get("generated")
                caption = example.get("caption", "")
                if args.flux:
                    prompt = ""
                elif args.validation_prompts and len(args.validation_prompts) > 0:
                    prompt = args.validation_prompts[min(local_idx, len(args.validation_prompts) - 1)]
                else:
                    prompt = caption if (isinstance(caption, str) and caption) else default_prompt

                sample_outputs = {
                    "base": {"preferred": None, "nonpreferred": None, "generated": None},
                    "trained": {"preferred": None, "nonpreferred": None, "generated": None},
                }

                inputs_for_pipe = {
                    "preferred": preferred_orig,
                    "nonpreferred": non_preferred_orig,
                    "generated": dataset_generated,
                }
                inputs_for_display = {
                    "preferred": preferred_orig,
                    "nonpreferred": non_preferred_orig,
                    "generated": dataset_generated,
                }
                metrics_ref_images = {
                    "preferred": preferred_orig,
                    "nonpreferred": non_preferred_orig,
                }
                size_lookup = {
                    name: (img.width, img.height) if img is not None else None
                    for name, img in inputs_for_pipe.items()
                }

                if args.flux:
                    size_candidates = [preferred_orig.size, non_preferred_orig.size]
                    if dataset_generated is not None:
                        size_candidates.append(dataset_generated.size)
                    target_size = choose_flux_resolution(size_candidates)
                    proc_pref, proc_nonpref, proc_generated = prepare_flux_images(
                        (preferred_orig, non_preferred_orig, dataset_generated),
                        target_size,
                        crop_u=0.5,
                        crop_v=0.5,
                        flip=False,
                    )
                    if proc_generated is None:
                        lut = _get_validation_lut(dataset_idx)
                        if lut is not None and proc_nonpref is not None:
                            proc_generated = apply_lut(proc_nonpref, lut)
                        else:
                            logger.debug(
                                "Validation sample %s lacks generated input after LUT fallback.",
                                dataset_idx,
                            )
                    inputs_for_pipe = {
                        "preferred": proc_pref,
                        "nonpreferred": proc_nonpref,
                        "generated": proc_generated,
                    }
                    inputs_for_display = {
                        "preferred": proc_pref or preferred_orig,
                        "nonpreferred": proc_nonpref or non_preferred_orig,
                        "generated": proc_generated,
                    }
                    # Align metrics with the resized/cropped references used for FLUX img2img inputs.
                    metrics_ref_images = {
                        "preferred": proc_pref or preferred_orig,
                        "nonpreferred": proc_nonpref or non_preferred_orig,
                    }
                    logger.debug(
                        "Flux validation metrics use preprocessed references at %s for dataset idx %s",
                        target_size,
                        dataset_idx,
                    )
                    size_lookup = {
                        name: (img.width, img.height) if img is not None else None
                        for name, img in inputs_for_pipe.items()
                    }

                eval_order = ["preferred", "nonpreferred", "generated"]
                source_sequence = [
                    (name, inputs_for_pipe[name]) for name in eval_order if inputs_for_pipe.get(name) is not None
                ]

                cached_base_outputs = {}
                if _baseline_cache_enabled and _baseline_cache_ready:
                    cached_base_outputs = _load_cached_baseline_outputs(dataset_idx)
                    for source_name, cached_image in cached_base_outputs.items():
                        if cached_image is None:
                            continue
                        sample_outputs["base"][source_name] = cached_image
                        pair_metrics = compute_pairwise_metrics(
                            cached_image,
                            metrics_ref_images["preferred"],
                            metrics_ref_images["nonpreferred"],
                        )
                        niqe_score = compute_niqe(cached_image)
                        metrics_plain = build_metric_record(pair_metrics, prefix=None, niqe_score=niqe_score)
                        group_key = f"base_{source_name}"
                        if source_name != "generated":
                            metrics_groups[group_key].append(metrics_plain)
                        table_rows.append((dataset_idx, "base", source_name, prompt, metrics_plain))

                seeds_for_sample = _validation_seeds_for_index(dataset_idx) if _baseline_cache_enabled else None

                pipelines_to_run = []
                if pipe_base is not None:
                    need_base_pipeline = True
                    if _baseline_cache_enabled and _baseline_cache_ready:
                        need_base_pipeline = any(
                            source_name not in cached_base_outputs for source_name, _ in source_sequence
                        )
                    if need_base_pipeline:
                        pipelines_to_run.append(("base", pipe_base))
                pipelines_to_run.append(("trained", pipe_trained))

                for pipeline_name, pipe in pipelines_to_run:
                    flux_callback = _build_flux_latent_dtype_callback(pipe)
                    for source_name, source_image in source_sequence:
                        if pipeline_name == "base" and _baseline_cache_enabled and _baseline_cache_ready:
                            if source_name in cached_base_outputs:
                                continue
                        if pipe is None:
                            continue
                        generator_seed = int(base_seed) + step * 1000 + local_idx * 10 + source_seed_offsets[source_name]
                        if seeds_for_sample is not None:
                            generator_seed = seeds_for_sample.get(
                                source_name, _validation_seed(dataset_idx, source_name)
                            )
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(generator_seed)

                        if args.flux:
                            size_tuple = size_lookup.get(source_name)
                            if size_tuple is None:
                                continue
                            image_for_pipe = _prepare_pipeline_image(pipe, source_image, size_tuple)
                            kwargs = dict(
                                image=image_for_pipe,
                                prompt="",
                                guidance_scale=args.validation_guidance_scale,
                                num_inference_steps=args.validation_inference_steps,
                                generator=generator,
                                height=size_tuple[1],
                                width=size_tuple[0],
                            )
                            if flux_callback is not None:
                                kwargs["callback_on_step_end"] = flux_callback
                                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                        else:
                            kwargs = dict(
                                image=source_image,
                                prompt=prompt,
                                strength=args.validation_strength,
                                guidance_scale=args.validation_guidance_scale,
                                num_inference_steps=args.validation_inference_steps,
                                generator=generator,
                            )

                        try:
                            with torch.no_grad():
                                out = pipe(**kwargs)
                        except Exception as e:
                            context = f"{pipeline_name}/{source_name}" if not args.flux else pipeline_name
                            logger.warning(
                                "Validation inference failed on sample %s (%s): %s",
                                dataset_idx,
                                context,
                                e,
                            )
                            continue

                        output_image = out.images[0] if hasattr(out, "images") and out.images else None
                        if output_image is None:
                            continue

                        sample_outputs[pipeline_name][source_name] = output_image
                        pair_metrics = compute_pairwise_metrics(
                            output_image,
                            metrics_ref_images["preferred"],
                            metrics_ref_images["nonpreferred"],
                        )
                        niqe_score = compute_niqe(output_image)
                        metrics_plain = build_metric_record(pair_metrics, prefix=None, niqe_score=niqe_score)
                        group_key = f"{pipeline_name}_{source_name}"
                        if not (pipeline_name == "base" and source_name == "generated"):
                            metrics_groups[group_key].append(metrics_plain)
                        table_rows.append((dataset_idx, pipeline_name, source_name, prompt, metrics_plain))

                if any(sample_outputs[p][s] is not None for p in sample_outputs for s in sample_outputs[p]):
                    samples_for_logging.append(
                        {
                            "dataset_index": dataset_idx,
                            "prompt": prompt,
                            "inputs": inputs_for_display,
                            "outputs": sample_outputs,
                        }
                    )

            if not metrics_groups:
                logger.info("Validation produced no samples; skipping wandb logging")
                return

            import wandb as _wandb

            metric_keys = [
                "psnr_pref",
                "psnr_nonpref",
                "psnr_margin",
                "ssim_pref",
                "ssim_nonpref",
                "ssim_margin",
                "delta_e_pref",
                "delta_e_nonpref",
                "delta_e_margin",
                "mse_pref",
                "mse_nonpref",
                "mse_margin",
                "niqe",
            ]

            log_payload = {}

            # Build per-sample metrics table for wandb
            table_columns = ["sample_id", "pipeline", "source", "prompt"] + metric_keys
            per_sample_table = _wandb.Table(columns=table_columns)
            for dataset_idx, pipeline_name, source_name, prompt, metrics_plain in table_rows:
                row = [dataset_idx, pipeline_name, source_name, prompt]
                for key in metric_keys:
                    row.append(metrics_plain.get(key))
                per_sample_table.add_data(*row)

            log_payload["validation/per_sample_metrics"] = per_sample_table

            # Aggregate metrics across samples
            summary_metrics = {}
            excluded_groups = {"base_preferred", "base_nonpreferred", "base_generated"}
            for group_key, records in metrics_groups.items():
                if group_key in excluded_groups:
                    continue
                aggregated = aggregate_metrics(records)
                summary_metrics.update(
                    {f"validation/{group_key}/{metric_key}": value for metric_key, value in aggregated.items()}
                )

            if summary_metrics:
                logger.info(
                    "Validation summary metrics: %s",
                    {k: round(v, 4) for k, v in sorted(summary_metrics.items())},
                )
            log_payload.update(summary_metrics)

            # Log qualitative images
            for display_idx, sample in enumerate(samples_for_logging):
                prompt = str(sample["prompt"])
                caption = f"step={step} | idx={sample['dataset_index']} | {prompt}"
                inputs_display = sample.get("inputs", {})
                outputs = sample["outputs"]

                preferred_input = inputs_display.get("preferred")
                nonpreferred_input = inputs_display.get("nonpreferred")
                generated_input = inputs_display.get("generated")

                base_outputs = outputs.get("base", {})
                trained_outputs = outputs.get("trained", {})

                base_pref = base_outputs.get("preferred")
                base_nonpref = base_outputs.get("nonpreferred")
                base_generated = base_outputs.get("generated")

                trained_pref = trained_outputs.get("preferred")
                trained_nonpref = trained_outputs.get("nonpreferred")
                trained_generated = trained_outputs.get("generated")

                grid_images = [
                    preferred_input,
                    nonpreferred_input,
                    generated_input,
                    base_pref,
                    base_nonpref,
                    base_generated,
                    trained_pref,
                    trained_nonpref,
                    trained_generated,
                ]

                if all(image is not None for image in grid_images):
                    try:
                        grid = make_image_grid(grid_images, rows=3, cols=3)
                        log_payload[f"validation/sample_{display_idx}_grid"] = _wandb.Image(grid, caption=caption)
                    except Exception as exc:
                        logger.debug(
                            "Failed to create validation grid for sample %s: %s",
                            sample["dataset_index"],
                            exc,
                        )
                else:
                    logger.debug(
                        "Skipping grid logging for validation sample %s due to missing images.",
                        sample["dataset_index"],
                    )

            _wandb.log(log_payload, step=step)
        except Exception as e:
            logger.warning(f"Failed to log validation metrics to wandb: {e}")
        finally:
            for module, was_training in modules_toggled:
                if was_training:
                    module.train()
            _cast_trainable_params_to_fp32(unet_trainable_fp32)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process and _baseline_cache_enabled:
        _ensure_baseline_cache_populated()

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = create_progress_bar(
        current_step=global_step,
        max_steps=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        description="Steps",
    )

        
    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        auxiliary_usage_accumulated = 0.0
        auxiliary_micro_accum = 0.0
        auxiliary_step_counter = 0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            current_beta_value = args.beta_dpo
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == 'dpo':
                    use_auxiliary_pair = False
                    selected_pixels = batch["pixel_values"]
                    if (
                        args.flux
                        and args.flux_aux_ratio > 0.0
                        and "pixel_values_generated" in batch
                        and random.random() < args.flux_aux_ratio
                    ):
                        selected_pixels = batch["pixel_values_generated"]
                        use_auxiliary_pair = True

                    # y_w and y_l were concatenated along channel dimension
                    feed_pixel_values = torch.cat(selected_pixels.chunk(2, dim=1))
                    # If using AIF then we haven't ranked yet so do so now
                    # Only implemented for BS=1 (assert-protected)
                    if args.choice_model:
                        if choice_model_says_flip(batch):
                            feed_pixel_values = feed_pixel_values.flip(0)
                    if args.flux and args.flux_aux_ratio > 0.0:
                        auxiliary_micro_accum += float(use_auxiliary_pair)
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]
                
                #### Diffusion Stuff ####
                flux_forward_kwargs = None
                if args.flux:
                    with torch.no_grad():
                        latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents.to(weight_dtype)
                    model_latents = (latents - vae_shift_factor) * vae_scaling_factor
                    noise = torch.randn_like(model_latents)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn(
                            (model_latents.shape[0], model_latents.shape[1], 1, 1), device=model_latents.device
                        )
                    if args.input_perturbation:
                        noise = noise + args.input_perturbation * torch.randn_like(noise)
                    bsz = model_latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_latents.device
                    ).long()
                    if args.train_method == 'dpo':
                        timesteps = timesteps.chunk(2)[0].repeat(2)
                        noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                    sigmas = get_flux_sigmas(timesteps, n_dim=model_latents.ndim, dtype=model_latents.dtype)
                    noisy_latents = (1.0 - sigmas) * model_latents + sigmas * noise
                    packed_latents = FluxPipeline._pack_latents(
                        noisy_latents,
                        batch_size=noisy_latents.shape[0],
                        num_channels_latents=noisy_latents.shape[1],
                        height=noisy_latents.shape[2],
                        width=noisy_latents.shape[3],
                    )
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        noisy_latents.shape[0],
                        noisy_latents.shape[2] // 2,
                        noisy_latents.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    prompt_embeds, pooled_prompt_embeds, text_ids = flux_encode_prompt_batch(
                        batch["clip_input_ids"],
                        batch["t5_input_ids"],
                        text_encoder_one,
                        text_encoder_two,
                        accelerator.device,
                        weight_dtype,
                    )
                    if args.train_method == 'dpo':
                        prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                        pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)
                    if getattr(unet.config, "guidance_embeds", False):
                        guidance_values = torch.full(
                            (noisy_latents.shape[0],),
                            args.guidance_scale,
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                    else:
                        guidance_values = None
                    flux_forward_kwargs = dict(
                        hidden_states=packed_latents,
                        timestep=timesteps / 1000,
                        guidance=guidance_values,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )
                    model_pred = unet(**flux_forward_kwargs)[0]
                    model_pred = FluxPipeline._unpack_latents(
                        model_pred,
                        height=model_latents.shape[2] * flux_vae_scale,
                        width=model_latents.shape[3] * flux_vae_scale,
                        vae_scale_factor=flux_vae_scale,
                    )
                    target = noise - model_latents
                else:
                    with torch.no_grad():
                        latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()
                    if 'refiner' in args.pretrained_model_name_or_path:
                        timesteps = timesteps % 200
                    elif 'turbo' in args.pretrained_model_name_or_path:
                        timesteps_0_to_3 = timesteps % 4
                        timesteps = 250 * timesteps_0_to_3 + 249

                    if args.train_method == 'dpo':
                        timesteps = timesteps.chunk(2)[0].repeat(2)
                        noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                    noisy_latents = noise_scheduler.add_noise(
                        latents, new_noise if args.input_perturbation else noise, timesteps
                    )
                    if args.sdxl:
                        with torch.no_grad():
                            if 'refiner' in args.pretrained_model_name_or_path:
                                add_time_ids = torch.tensor([
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    6.0,
                                ], dtype=weight_dtype, device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                            else:
                                add_time_ids = torch.tensor([
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    args.resolution,
                                    args.resolution,
                                ], dtype=weight_dtype, device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                            prompt_batch = encode_prompt_sdxl(
                                batch,
                                text_encoders,
                                tokenizers,
                                args.proportion_empty_prompts,
                                caption_column='caption',
                                is_train=True,
                            )
                        if args.train_method == 'dpo':
                            prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                            prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                        unet_added_conditions = {
                            "time_ids": add_time_ids,
                            "text_embeds": prompt_batch["pooled_prompt_embeds"],
                        }
                    else:
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        if args.train_method == 'dpo':
                            encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

                    assert noise_scheduler.config.prediction_type == "epsilon"
                    target = noise

                    model_batch_args = (
                        noisy_latents,
                        timesteps,
                        prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                    )
                    added_cond_kwargs = unet_added_conditions if args.sdxl else None
                    model_pred = unet(
                        *model_batch_args,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.train_method == 'dpo':
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
                    model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        if args.flux:
                            ref_out = ref_unet(**flux_forward_kwargs)[0]
                            ref_pred = FluxPipeline._unpack_latents(
                                ref_out,
                                height=model_latents.shape[2] * flux_vae_scale,
                                width=model_latents.shape[3] * flux_vae_scale,
                                vae_scale_factor=flux_vae_scale,
                            ).detach()
                        else:
                            ref_pred = ref_unet(
                                *model_batch_args,
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()    
                        
                    if beta_schedule_fn is not None:
                        current_beta_value = beta_schedule_fn(global_step)
                    scale_term = -0.5 * current_beta_value
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == 'dpo':
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    _cast_trainable_grads_to_fp32(unet_trainable_fp32)
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                if args.train_method == 'dpo' and args.flux and args.flux_aux_ratio > 0.0:
                    step_aux_rate = auxiliary_micro_accum / float(args.gradient_accumulation_steps)
                    auxiliary_usage_accumulated += step_aux_rate
                    auxiliary_step_counter += 1
                    auxiliary_micro_accum = 0.0
                else:
                    auxiliary_micro_accum = 0.0
                progress_bar.update(1)
                global_step += 1
                # Scalar logging controlled by interval
                if args.log_scalar_steps > 0 and (global_step % args.log_scalar_steps == 0):
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    if args.train_method == 'dpo':
                        accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                        accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                        accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                        accelerator.log({"beta_dpo": float(current_beta_value)}, step=global_step)
                        if args.flux and args.flux_aux_ratio > 0.0:
                            mean_aux_rate = (
                                auxiliary_usage_accumulated / float(auxiliary_step_counter)
                                if auxiliary_step_counter > 0
                                else 0.0
                            )
                            accelerator.log({"auxiliary_step_rate": mean_aux_rate}, step=global_step)
                # Validation (baseline vs trained img2img)
                _run_validation_if_needed(global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0
                auxiliary_usage_accumulated = 0.0
                auxiliary_step_counter = 0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_acc
                logs["beta_dpo"] = current_beta_value
                if args.flux and args.flux_aux_ratio > 0.0:
                    logs["auxiliary_pair"] = float(use_auxiliary_pair)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.flux and args.flux_use_lora:
            unet.save_lora_adapter(args.output_dir, adapter_name=flux_lora_adapter_name or "flux_lora")
        elif args.flux:
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unet,
                vae=vae,
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.save_pretrained(args.output_dir)
        elif args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision, torch_dtype=weight_dtype
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)


    accelerator.end_training()


if __name__ == "__main__":
    main()
