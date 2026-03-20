#!/bin/bash

# 设置CUDA设备号
export CUDA_VISIBLE_DEVICES=6

# Python脚本路径
PYTHON_SCRIPT="/mnt/data/project_kyh/Anchor-master/inference/pix2pix-zero.py"

SOURCE_MODALITY=OPT
TARGET_MODALITY=NIR
# 参数配置
JSONL_PATH="/mnt/data/project_kyh/Anchor-master/data/metadata/Diffusion/DFC2019_small/test/msi.jsonl"
BASE_MODEL_PATH="/mnt/data/project_kyh/Anchor-master/weight/RS-Diffusion/sd-DFC2019_small-model/rgb/checkpoint-5000"
BASE_MODEL_PATH_2="/mnt/data/project_kyh/Anchor-master/weight/RS-Diffusion/sd-DFC2019_small-model/msi/checkpoint-5000"
CONTROLNET_PATH="/mnt/data/project_kyh/Anchor-master/weight/RS-ControlNet/DFC2019_small_OPT2NIR/checkpoint-15000/controlnet"
RESULTS_FOLDER="/mnt/data/project_kyh/Anchor-master/result/pix2pix_DFC2019_small_${SOURCE_MODALITY}2${TARGET_MODALITY}"

NUM_DDIM_STEPS=50
XA_GUIDANCE=0.15
NEGATIVE_GUIDANCE_SCALE=7.5
USE_FLOAT_16="--use_float_16"  # 有则填，没有则空

mkdir -p $RESULTS_FOLDER

python3 $PYTHON_SCRIPT \
    --jsonl_path $JSONL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --base_model_path_2 $BASE_MODEL_PATH_2 \
    --controlnet_path $CONTROLNET_PATH \
    --results_folder $RESULTS_FOLDER \
    --source_modality $SOURCE_MODALITY \
    --target_modality $TARGET_MODALITY \
    --num_ddim_steps $NUM_DDIM_STEPS \
    --xa_guidance $XA_GUIDANCE \
    --negative_guidance_scale $NEGATIVE_GUIDANCE_SCALE \
    $USE_FLOAT_16
