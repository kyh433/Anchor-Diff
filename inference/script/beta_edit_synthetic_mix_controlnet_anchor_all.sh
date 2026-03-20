#!/bin/bash

# 设置CUDA设备号
export CUDA_VISIBLE_DEVICES=2

# Python脚本路径
PYTHON_SCRIPT="/mnt/data/project_kyh/MultimodalityGeneration25/src/beta_edit_synthetic_mix_controlnet_anchor_all.py"


# 参数配置
JSONL_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan_anchor/test/s2_ir.jsonl"
BASE_MODEL_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_nat/checkpoint-5000"
BASE_MODEL_PATH_2="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_ir/checkpoint-5000"
CONTROLNET_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model-controlnet_anchor/s2_ir_anchor_by_nat/checkpoint-20000/controlnet"
MMVAE_PATH="/mnt/data/project_kyh/MultimodalityGeneration25/runs/mmvm_vae_hunan_v4_512_pair/best.pt"
RESULTS_FOLDER="anchor_output/s2_ir_test_cp-20000_controlnet"
NUM_DDIM_STEPS=20
XA_GUIDANCE=0.15
NEGATIVE_GUIDANCE_SCALE=7.5
USE_FLOAT_16="--use_float_16"  # 有则填，没有则空

mkdir -p $RESULTS_FOLDER

python3 $PYTHON_SCRIPT \
    --jsonl_path $JSONL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --base_model_path_2 $BASE_MODEL_PATH_2 \
    --controlnet_path $CONTROLNET_PATH \
    --mmvae_path $MMVAE_PATH \
    --results_folder $RESULTS_FOLDER \
    --num_ddim_steps $NUM_DDIM_STEPS \
    --xa_guidance $XA_GUIDANCE \
    --negative_guidance_scale $NEGATIVE_GUIDANCE_SCALE \
    $USE_FLOAT_16
