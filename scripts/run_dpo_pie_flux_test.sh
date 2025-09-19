#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/_hf_cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

MODEL_REPO="black-forest-labs--FLUX.1-Kontext-dev"
MODEL_HASH_FILE="${HUGGINGFACE_HUB_CACHE}/models--${MODEL_REPO}/refs/main"

if [[ ! -f "${MODEL_HASH_FILE}" ]]; then
  echo "[error] Expected local checkpoint ref at ${MODEL_HASH_FILE} but it was not found." >&2
  echo "        Run 'huggingface-cli download ${MODEL_REPO/--//}' to populate the cache." >&2
  exit 1
fi

MODEL_HASH=$(<"${MODEL_HASH_FILE}")
MODEL_DIR="${HUGGINGFACE_HUB_CACHE}/models--${MODEL_REPO}/snapshots/${MODEL_HASH}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[error] Resolved snapshot directory ${MODEL_DIR} does not exist." >&2
  exit 1
fi

echo "Using local FLUX checkpoint at ${MODEL_DIR}" >&2

accelerate launch --mixed_precision "bf16" --num_processes 1 --num_machines 1 --dynamo_backend "no" train.py \
  --model_arch flux \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --mixed_precision bf16 \
  --custom_data_root "datasets" \
  --user_json_path "datasets/responses/user_response_example.json" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 10000 \
  --learning_rate 2e-4 \
  --beta_dpo 5000 \
  --beta_dpo_start 1000 \
  --beta_dpo_schedule linear \
  --beta_dpo_schedule_steps 10000 \
  --num_validation_images 4 \
  --validation_inference_steps 28 \
  --validation_guidance_scale 1.0 \
  --validation_steps 100 \
  --validation_use_baseline_cache \
  --checkpointing_steps 100 \
  --flux_use_lora \
  --flux_lora_r 32 \
  --flux_lora_alpha 32 \
  --flux_lora_dropout 0.05 \
  --flux_aux_ratio 0.2 \
  --flux_aux_inference_steps 10 \
  --flux_aux_lut_dir "datasets/LUTs" \
  --flux_aux_lut_order "red_fastest" \
  --disable_xformers \
  --output_dir "out-flux-test" \
  --report_to "wandb"
