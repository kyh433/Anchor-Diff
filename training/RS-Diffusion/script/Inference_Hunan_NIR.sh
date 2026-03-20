
#!/bin/bash
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO
export MODEL_NAME="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_ir/checkpoint-5000/"
export METADATA="/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan/test/s2_ir.jsonl"
export OUTPUT_DIR="/mnt/data/project_kyh/MultimodalityGeneration25/output/alpha_Hunan/Fine-tuned_Result/s2_ir"

accelerate launch --config_file=/home/isalab304/.cache/huggingface/accelerate/card4.yaml --num_processes 1 --mixed_precision="fp16" \
   Finetune_Inference.py \
  --finetune_model=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --prompt_jsonl=$METADATA \
  --num_variants=3 \
  --inference_steps=30 \
  --guidance_scale=7.5 \
  --seed_start=42 \
  