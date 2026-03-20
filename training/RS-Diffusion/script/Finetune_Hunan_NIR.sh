
#!/bin/bash
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO

export MODEL_NAME="/mnt/data/project_kyh/weight/lllyasviel/stable-diffusion-v1-5"
export TRAIN_DIR="/mnt/data/project_kyh/MultimodalityGeneration25/traindata/"
export METADATA="split_by_modality_Hunan/train/s2_ir.jsonl"
export OUTPUT_DIR="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_ir"

accelerate launch --config_file=/home/isalab304/.cache/huggingface/accelerate/card45.yaml \
  --main_process_port 29503 --num_processes 2 --mixed_precision="fp16" --multi_gpu \
  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --metadata_name=$METADATA \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --image_column="file_name" \
  --caption_column="text" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --tracker_project_name="text2image-fine-tune-Hunan" \
  --validation_prompts "The image is a satellite image of a rural area with a few small towns and roads" \
  "The image is a satellite image of a large urban area, likely a city or town, with a mix of residential and commercial buildings" \
  --report_to="wandb"