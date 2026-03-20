#!/usr/bin/env bash
set -euo pipefail
# export CUDA_VISIBLE_DEVICES=1
export MODALITY="s2_ir"
export MODEL_NAME="/mnt/data/project_kyh/weight/lllyasviel/stable-diffusion-v1-5"
export FINETUNE_NAME="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/${MODALITY}/checkpoint-5000"

export TRAIN_DIR="/mnt/data/project_kyh/MultimodalityGeneration25/traindata"
export JSONL_DIR="split_by_modality_Hunan_anchor_edge/train/${MODALITY}_anchor_by_nat.jsonl"
export OUTPUT_DIR="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model-controlnet_anchor_edge_nat/${MODALITY}"

# python ./code_anchor/alpha_06.02_train_controlnet_anchor2other.py \

accelerate launch \
  --config_file=/home/isalab304/.cache/huggingface/accelerate/card01.yaml \
  --main_process_port 29505 \
  --num_processes 2 \
  --mixed_precision fp16 \
  --multi_gpu \
  ./code_anchor/alpha_06.02_train_controlnet_anchor2other.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --finetune_path="$FINETUNE_NAME" \
  --train_data_dir="$TRAIN_DIR" \
  --jsonl_dir="$JSONL_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution 512 \
  --learning_rate 1e-5 \
  --conditioning_mode npy \
  --conditioning_channels 4 \
  --conditioning_npy_norm minmax \
  --conditioning_npy_interp bilinear \
  --validation_image \
    "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural_fusion_edge/natural_15184.npy" \
  --validation_prompt \
    "The image is a satellite image of a river with a bridge spanning it, surrounded by a forested landscape" \
  --train_batch_size 12 \
  --tracker_project_name "controlnet-Hunan-anchor-edge" \
  --report_to wandb \
  --max_train_steps 50000 \
  --checkpointing_steps 1000 \
  --dataloader_num_workers 8 \
  --resume_from_checkpoint "/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model-controlnet_anchor_edge_nat/s2_ir/checkpoint-20000"
