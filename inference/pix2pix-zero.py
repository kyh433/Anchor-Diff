import os
import argparse
import json
import torch
from PIL import Image
from diffusers import DDIMScheduler, UniPCMultistepScheduler, StableDiffusionPipeline

from utils.edit_pipeline import EditingPipeline_control
from utils.base_pipeline import BasePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("可见设备：", torch.cuda.device_count())
print("使用的设备：", torch.cuda.current_device())
print("设备名称：", torch.cuda.get_device_name(0))

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, required=True)
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--source_modality', type=str, default='OPT')
    parser.add_argument('--target_modality', type=str, default='SAR')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--base_model_path_2', type=str, required=True)
    parser.add_argument('--controlnet_path', type=str, required=True)
    parser.add_argument('--xa_guidance', default=0.15, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_folder, args.source_modality), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, args.target_modality), exist_ok=True)

    torch_dtype = torch.float16 if args.use_float_16 else torch.float32
    generator = torch.manual_seed(0)
    x = torch.randn((1, 4, 64, 64), device=device)

    pipe1 = StableDiffusionPipeline.from_pretrained(args.base_model_path, torch_dtype=torch.float32).to(device)

    # 保留 --controlnet_path 接口以兼容现有 .sh，但不再加载/使用 ControlNet。
    pipe2 = StableDiffusionPipeline.from_pretrained(args.base_model_path_2, torch_dtype=torch_dtype).to(device)

    pipe = EditingPipeline_control(model1=pipe1, model2=pipe2)

    pipe1.scheduler = UniPCMultistepScheduler.from_config(pipe1.scheduler.config)
    pipe2.scheduler = UniPCMultistepScheduler.from_config(pipe2.scheduler.config)
    pipe1.enable_model_cpu_offload()
    pipe2.enable_xformers_memory_efficient_attention()
    pipe2.safety_checker = None

    for item in read_jsonl(args.jsonl_path):
        prompt = item['text']
        file_name = os.path.basename(item['file_name']).replace('.png', '')

        print(f"正在处理：{file_name}，Prompt: {prompt}")
        rec_pil, edit_pil, edge_pil = pipe(
            prompt,
            num_inference_steps=args.num_ddim_steps,
            x_in=x,
            guidance_amount=args.xa_guidance,
            guidance_scale=args.negative_guidance_scale,
            negative_prompt='',
            generator=generator,
            only_sample=False,
        )

        edit_pil[0].save(os.path.join(args.results_folder, args.target_modality, f"{file_name}.png"))
        rec_pil[0].save(os.path.join(args.results_folder, args.source_modality, f"{file_name}.png"))
