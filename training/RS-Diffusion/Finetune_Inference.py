import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
def load_prompt_mapping(jsonl_path):
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            file_name = obj["file_name"]  # e.g., "Processed_Hunan_Dataset/test/s2/natural/natural_11624.png"
            image_id = os.path.splitext(os.path.basename(file_name))[0]  # e.g., "natural_11624"
            prompt = obj["text"]
            mapping[image_id] = prompt
    return mapping  # { "natural_11624": "The image appears to be..." }

def setup_pipeline(finetune_model, base_model="/mnt/data/project_kyh/weight/lllyasviel/stable-diffusion-v1-5", dtype=torch.float16):
    # 1. 加载原始 Stable Diffusion 模型（用作 VAE、tokenizer 等组件来源）
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype).to("cuda")
    unet_path = os.path.join(finetune_model, "unet")
    # 2. 替换微调后的 UNet
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, 
        subfolder="",  # 没有嵌套子目录
        torch_dtype=dtype
    ).to("cuda") 
    pipe.unet = unet
    pipe.safety_checker = None
    # 3. 启用 xformers（可选）
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"⚠️ Failed to enable xformers: {e}")

    return pipe

@torch.inference_mode()
def generate_images(pipe, prompt, seeds, steps, guidance):
    images = []
    for seed in seeds:
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
        images.append((seed, image))
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_jsonl", type=str, required=True, help="JSON file with {image_id: prompt}")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--finetune_model", type=str, default="/mnt/data/project_kyh/weight/lllyasviel/stable-diffusion-v1-5")
    parser.add_argument("--num_variants", type=int, default=3, help="Number of images per prompt")
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed_start", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompt_mapping(args.prompt_jsonl)
    pipe = setup_pipeline(finetune_model=args.finetune_model)

    for img_id, prompt in tqdm(prompts.items(), desc="Generating"):
        seeds = [args.seed_start + i for i in range(args.num_variants)]
        results = generate_images(pipe, prompt, seeds, args.inference_steps, args.guidance_scale)
        for seed, image in results:
            filename = f"{img_id}_seed{seed}.png"
            save_path = os.path.join(args.output_dir, filename)
            image.save(save_path)

if __name__ == "__main__":
    main()
