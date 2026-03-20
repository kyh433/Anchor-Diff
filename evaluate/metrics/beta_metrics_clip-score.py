import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ====== 1. Load images from folder (same as before) ======
def load_images_from_folder(folder_path, file_type=None):
    images = []
    filenames = []
    image_files = sorted(os.listdir(folder_path))

    for image_file in image_files:
        if image_file.endswith(".png"):
            if file_type and file_type in image_file:
                img_path = os.path.join(folder_path, image_file)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                filenames.append(image_file)
            elif not file_type:
                img_path = os.path.join(folder_path, image_file)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                filenames.append(image_file)
    return images, filenames


# ====== 2. Load text jsonl (your version) ======
def load_text_from_jsonl(text_file):
    with open(text_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    texts = [entry['text'] for entry in data]
    return texts


# ====== 3. Load CLIP from local weights (your required path) ======
def get_clip_model():
    model = CLIPModel.from_pretrained("/mnt/data/project_kyh/weight/openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("/mnt/data/project_kyh/weight/openai/clip-vit-large-patch14")
    return model, processor


# ====== 4. Compute CLIPScore for paired (image, text) ======
@torch.no_grad()
def compute_clipscore_for_pairs(images, texts, model, processor, device, batch_size=32):
    """
    images: list[PIL.Image]
    texts: list[str]
    len(images) must equal len(texts)
    returns (mean_score, per_sample_scores)
    """
    assert len(images) == len(texts), \
        f"Number of images ({len(images)}) must equal number of texts ({len(texts)})"

    model.eval()
    scores = []
    n = len(images)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_images = images[start:end]
        batch_texts = texts[start:end]

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs = model(**inputs)
        img_feat = outputs.image_embeds            # (B, D)
        txt_feat = outputs.text_embeds             # (B, D)

        # normalize -> cosine
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        batch_scores = (img_feat * txt_feat).sum(dim=-1)  # (B,)

        scores.append(batch_scores.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    return float(scores.mean()), scores


# ====== 5. Main (keep nat / vh groups) ======
def compute_clipscore(real_nat_folder, real_vh_folder, generated_folder, text_file, device="cuda"):
    """
    说明：
    - text_file 中 prompts 的顺序 = 生成图顺序
    - 默认前半对应 reconstruct(nat)，后半对应 edit(vh)
      如果你的 jsonl 只对应 nat 或 vh，请自行调整切分方式。
    """

    # # 你的 val 生成路径（保持你原来的写法）
    # val_nat_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural"
    # val_vh_folder  = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh"

    generated_images_reconstruct, gen_nat_names = load_images_from_folder(real_nat_folder)
    generated_images_edit, gen_vh_names = load_images_from_folder(real_vh_folder)

    # Load CLIP
    clip_model, clip_processor = get_clip_model()
    clip_model = clip_model.to(device)

    # Load prompts directly from jsonl (your requirement)
    texts = load_text_from_jsonl(text_file)

    # ====== 按“生成顺序”直接切分 ======
    n_nat = len(generated_images_reconstruct)
    n_vh  = len(generated_images_edit)

    nat_texts = texts
    vh_texts  = texts

    # nat / reconstruct CLIPScore
    nat_clipscore_mean, _ = compute_clipscore_for_pairs(
        generated_images_reconstruct, nat_texts,
        clip_model, clip_processor, device
    )
    print(f"nat CLIPScore: {nat_clipscore_mean:.4f}")

    # vh / edit CLIPScore
    vh_clipscore_mean, _ = compute_clipscore_for_pairs(
        generated_images_edit, vh_texts,
        clip_model, clip_processor, device
    )
    print(f"vh CLIPScore: {vh_clipscore_mean:.4f}")

    return nat_clipscore_mean, vh_clipscore_mean


# ====== 6. Example usage ======
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_nat_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s2/natural"
    real_vh_folder  = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s1/vh"
    generated_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/beta_output/natPCA2s1_vh_attention"

    # 你的 jsonl prompt 地址（按你给的）
    text_file = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan/test/s2_nat.jsonl"

    compute_clipscore(real_nat_folder, real_vh_folder, generated_folder, text_file, device=device)
