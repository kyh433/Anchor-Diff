import os
import sys
import json
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils import load_image
import torch.nn.functional as F

# =========================
#  MMVM-VAE Anchor Extractor
# =========================
class MMVMAAnchorExtractor:
    """
    On-the-fly MMVM-VAE scene-anchor feature extractor.

    Output:
      - torch.FloatTensor of shape [B, z_ch, h, w] (mu_map)
    """

    def __init__(
        self,
        ckpt_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        size: int = 256,
        modality: str = "natural",
        norm: str = "none",
    ):
        self.ckpt_path = ckpt_path
        self.size = int(size)
        self.modality = modality
        self.norm = (norm or "none").lower()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 让脚本在 src/ 下运行时也能 import 到项目根目录的 mmvae/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 兼容两种导入路径
        try:
            from mmvae.mmvm_multimodal_anchor_vae_v4 import build_model, load_ckpt
        except Exception:
            from mmvm_multimodal_anchor_vae_v4 import build_model, load_ckpt  # type: ignore

        self._build_model = build_model
        self._load_ckpt = load_ckpt

        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        train_args_dict = ckpt.get("args", None)
        if train_args_dict is None:
            raise RuntimeError("MMVM checkpoint has no 'args'. Cannot rebuild model reliably.")

        class _A:
            pass

        train_args = _A()
        for k, v in train_args_dict.items():
            setattr(train_args, k, v)

        model = self._build_model(train_args)

        # 兼容不同 load_ckpt 签名
        try:
            _ = self._load_ckpt(self.ckpt_path, model, optimizer=None, scaler=None, map_location="cpu")
        except TypeError:
            _ = self._load_ckpt(self.ckpt_path, model)

        return model

    @staticmethod
    def _pil_to_tensor_3ch(img: Image.Image, size: int) -> torch.FloatTensor:
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), resample=Image.BICUBIC)

        arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3] in [0,1]
        arr = np.transpose(arr, (2, 0, 1))  # [3,H,W]
        return torch.from_numpy(arr)

    @staticmethod
    def _apply_norm(x: torch.Tensor, norm: str, eps: float = 1e-6) -> torch.Tensor:
        norm = (norm or "none").lower()
        if norm == "none":
            return x
        if norm == "minmax":
            mn = x.amin(dim=(1, 2, 3), keepdim=True)
            mx = x.amax(dim=(1, 2, 3), keepdim=True)
            return (x - mn) / (mx - mn + eps)
        if norm == "zscore":
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            return (x - mean) / (std + eps)
        raise ValueError(f"Unknown mmvae norm: {norm}")

    @torch.no_grad()
    def extract_mu_map(self, image: Union[Image.Image, List[Image.Image]]) -> torch.FloatTensor:
        if isinstance(image, list):
            xs = [self._pil_to_tensor_3ch(im, self.size) for im in image]
            x = torch.stack(xs, dim=0)  # [B,3,H,W]
        else:
            x = self._pil_to_tensor_3ch(image, self.size).unsqueeze(0)  # [1,3,H,W]

        x = x.to(self.device, dtype=torch.float32)

        # Expect: mu, logvar = model.encode(modality, x)
        try:
            mu, _logvar = self.model.encode(self.modality, x)
        except TypeError:
            out = self.model.encode(x, self.modality)
            mu = out[0] if isinstance(out, (tuple, list)) else out

        mu = mu.detach().float()
        mu = self._apply_norm(mu, self.norm)
        return mu

def load_condition_npy(path: str, resolution: int = 512, norm: str = "none", interp: str = "bilinear"):
    """Load a conditioning feature saved as .npy and resize to (C, resolution, resolution).

    Expected npy shapes:
      - (C, H, W)
      - (H, W)          -> will become (1, H, W)
      - (H, W, C)       -> will be transposed to (C, H, W)
    """
    arr = np.load(path)
    x = torch.from_numpy(arr).float()  # (C,H,W)
    if norm == "minmax":
        mn = x.amin(dim=(1, 2), keepdim=True)
        mx = x.amax(dim=(1, 2), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)
    elif norm == "zscore":
        mu = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        x = (x - mu) / std

    # Resize to training resolution
    x = x.unsqueeze(0)  # (1,C,H,W)
    x = F.interpolate(
        x,
        size=(resolution, resolution),
        mode=interp,
        align_corners=False if interp in ("bilinear", "bicubic") else None,
    )
    return x.squeeze(0)  # (C,res,res)


def condition_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """For logging only: convert a (C,H,W) tensor to a displayable PIL RGB image."""
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    # Use mean across channels for visualization
    if x.shape[0] > 1:
        m = x.mean(dim=0)
    else:
        m = x[0]
    m = m - m.min()
    m = m / (m.max() + 1e-6)
    m = (m.numpy() * 255.0).astype("uint8")
    return Image.fromarray(m).convert("RGB")
# =========================
#  Main
# =========================
def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # 路径设置
    jsonl_path = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan/test/s2_nat.jsonl"
    base_model_path = "/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_nat/checkpoint-5000"
    base_model_path_2 = "/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model/s2_ir/checkpoint-5000"
    controlnet_path = "/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model-controlnet_anchor_edge_nat/s2_ir/checkpoint-20000/controlnet"
    # controlnet_path = "/mnt/data/project_kyh/MultimodalityGeneration25/weight/sd-Hunan-model-controlnet_nat/s2_ir/checkpoint-10000/controlnet"
    image_root = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata"


    save_dir = "./anchor_output/debug_nat2ir_controlnet_edge"
    os.makedirs(save_dir, exist_ok=True)

  

    # 模型加载
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path_2, controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe_ori = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32).to(device)
    pipe.safety_checker = None
    pipe_ori.safety_checker = None

    # 批量处理
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        try:
            data = json.loads(line)
            image_path = os.path.join(image_root, data["file_name"])
            prompt = data["text"]
            img_name = os.path.basename(data["file_name"])[:-4]

            # if "natural_11624" not in img_name:
            #     continue

            # 原始图像推理生成（reconstruction）
            input_image = load_image(image_path).convert("RGB")
            generator = torch.manual_seed(0)
            
            # opt_image = pipe_ori(prompt, num_inference_steps=20, generator=generator).images[0]
            opt_image_path = os.path.join(save_dir, f"{img_name}_reconstruction.png")
            input_image.save(opt_image_path)
            # control_cond = opt_image

            # prompt = "The image is a satellite image of a river with a bridge spanning it, surrounded by a forested landscape"
            # control_cond_path = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural_fusion_edge/natural_15184.npy"
            control_cond_path = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s2/natural_fusion_edge/"+data["file_name"].split('/')[-1].replace('.png','.npy')
            control_input = load_condition_npy(
                control_cond_path,
                resolution=512,
                norm='minmax',
                interp='bilinear',
            ).to(device, dtype=torch.float32)
            # pipeline expects batched tensor for torch input
            control_cond = control_input.unsqueeze(0)
            validation_image_vis = condition_tensor_to_pil(control_input)
            validation_image_vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
            validation_image_vis.save(validation_image_vis_path)

            # # ✅ 关键修复：用实例对象调用
            # control_cond = mmvae_extractor.extract_mu_map(opt_image)
            # control_cond = control_cond.to(device=device, dtype=torch.float16)

            # height =  pipe_ori.unet.config.sample_size * pipe_ori.vae_scale_factor
            # width =  pipe_ori.unet.config.sample_size * pipe_ori.vae_scale_factor
            # control_cond = F.interpolate(control_cond, size=(height, width), mode="bilinear", align_corners=False)
            # torch.cuda.empty_cache()

            # ControlNet 推理
            torch.cuda.empty_cache()
            generator = torch.manual_seed(0)
            output_image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_cond).images[0]
            output_image_path = os.path.join(save_dir, f"{img_name}_edit.png")
            output_image.save(output_image_path)

            print(f"[{idx+1}/{len(lines)}] Processed: {image_path}")

        except Exception as e:
            print(f"Error processing line {idx}: {e}")


if __name__ == "__main__":
    main()
