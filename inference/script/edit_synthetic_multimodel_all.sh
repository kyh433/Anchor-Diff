#!/bin/bash

# 设置CUDA设备号
export CUDA_VISIBLE_DEVICES=3

# Python脚本路径
PYTHON_SCRIPT="./src/edit_synthetic_multimodel_all.py"


# 参数配置
# JSONL_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan/test/s1_vh.jsonl"
# MODEL_PATH_1="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_nat/checkpoint-5000"
# MODEL_PATH_2="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s1_vh/checkpoint-5000"

JSONL_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_FUSAR-MaP/test/sar.jsonl"
MODEL_PATH_1="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-FUSAR-MaP-model/opt/checkpoint-15000"
MODEL_PATH_2="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-FUSAR-MaP-model/sar/checkpoint-15000"
RESULTS_FOLDER="beta_output/OPT2SAR_pix2pix_XA_GUIDANCE0.3"
NUM_DDIM_STEPS=50
XA_GUIDANCE=0.3
NEGATIVE_GUIDANCE_SCALE=7.5
USE_FLOAT_16="--use_float_16"  # 有则填，没有则空

mkdir -p $RESULTS_FOLDER

python3 $PYTHON_SCRIPT \
    --jsonl_path $JSONL_PATH \
    --model_path1 $MODEL_PATH_1 \
    --model_path2 $MODEL_PATH_2 \
    --results_folder $RESULTS_FOLDER \
    --num_ddim_steps $NUM_DDIM_STEPS \
    --xa_guidance $XA_GUIDANCE \
    --negative_guidance_scale $NEGATIVE_GUIDANCE_SCALE \
    $USE_FLOAT_16
