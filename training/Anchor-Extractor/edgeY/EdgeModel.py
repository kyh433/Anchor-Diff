import os
import glob
import re

import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# Utils: deterministic
# =========================================================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset: basename matching to avoid modality misalignment
# =========================================================
def get_preprocessing_transform(size=(512, 512)):
    # Percentile contrast stretching per channel (same as your original)
    def stretch(img):
        img_np = np.array(img).astype(np.float32)
        for i in range(3):
            p2 = np.percentile(img_np[..., i], 2)
            p98 = np.percentile(img_np[..., i], 98)
            if p98 > p2:
                img_np[..., i] = (img_np[..., i] - p2) / (p98 - p2)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    return transforms.Compose([
        transforms.Lambda(stretch),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])



class MultiModalDataset(Dataset):
    """
    Pair by extracted {id} from filenames like:
      natural_{id}.png, infrared_{id}.png, sar_{id}.png

    It will:
      - extract id via regex
      - build maps: id -> path
      - take intersection ids
    """
    def __init__(self, natural_dir, infrared_dir, sar_dir, transform=None):
        self.transform = transform

        nat_paths = glob.glob(os.path.join(natural_dir, "*.png"))
        ir_paths  = glob.glob(os.path.join(infrared_dir, "*.png"))
        sar_paths = glob.glob(os.path.join(sar_dir, "*.png"))

        # pattern: "<modality>_<id>.png"  (id can contain digits/letters, and underscores too if needed)
        # Example matches: natural_0001.png -> id=0001
        # If your id contains underscores, change ([^.]*) to something else.
        pat = re.compile(r"^(natural|infrared|vh)_(.+)\.png$", re.IGNORECASE)

        def build_id_map(paths, expected_mod):
            m = {}
            for p in paths:
                name = os.path.basename(p)
                mm = pat.match(name)
                if mm is None:
                    continue
                mod, id_ = mm.group(1).lower(), mm.group(2)
                if mod != expected_mod:
                    continue
                m[id_] = p
            return m

        nat_map = build_id_map(nat_paths, "natural")
        ir_map  = build_id_map(ir_paths,  "infrared")
        sar_map = build_id_map(sar_paths, "vh")

        keys = sorted(list(set(nat_map.keys()) & set(ir_map.keys()) & set(sar_map.keys())))
        if len(keys) == 0:
            # helpful debug info
            nat_any = next(iter(nat_map.keys()), None)
            ir_any  = next(iter(ir_map.keys()), None)
            sar_any = next(iter(sar_map.keys()), None)
            raise RuntimeError(
                "No matched IDs across natural/infrared/sar folders.\n"
                f"Parsed counts: natural={len(nat_map)}, infrared={len(ir_map)}, sar={len(sar_map)}\n"
                f"Example IDs: natural={nat_any}, infrared={ir_any}, sar={sar_any}\n"
                "Check filename pattern '<modality>_<id>.png' and folder contents."
            )

        self.samples = [(nat_map[k], ir_map[k], sar_map[k], k) for k in keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nat_path, ir_path, sar_path, id_ = self.samples[idx]

        nat = Image.open(nat_path).convert("RGB")
        ir  = Image.open(ir_path).convert("RGB")
        sar = Image.open(sar_path).convert("RGB")

        if self.transform:
            nat = self.transform(nat)
            ir  = self.transform(ir)
            sar = self.transform(sar)

        # 这里 filename 我用统一 id 输出，避免你后面保存时出现 natural_/infrared_ 前缀混乱
        return {"natural": nat, "infrared": ir, "sar": sar, "filename": f"{id_}.png"}



def get_loader(nat_dir, ir_dir, sar_dir, batch_size=4, shuffle=True, num_workers=2):
    tfm = get_preprocessing_transform()
    ds = MultiModalDataset(nat_dir, ir_dir, sar_dir, tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# =========================================================
# Model: lightweight UNet per modality + attention fusion at bottleneck
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


class EncoderUNetLite(nn.Module):
    """
    returns:
      f1: [B, C1, H,   W  ]  (shallow)
      f2: [B, C2, H/2, W/2]
      f3: [B, C3, H/4, W/4]  (bottleneck)
    """
    def __init__(self, in_channels=3, c1=32, c2=64, c3=128):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, c1),
            ConvBNReLU(c1, c1),
        )
        self.down1 = ConvBNReLU(c1, c2, s=2)  # /2
        self.stage2 = nn.Sequential(
            ConvBNReLU(c2, c2),
            ConvBNReLU(c2, c2),
        )
        self.down2 = ConvBNReLU(c2, c3, s=2)  # /4
        self.stage3 = nn.Sequential(
            ConvBNReLU(c3, c3),
            ConvBNReLU(c3, c3),
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        return f1, f2, f3


class AttentionFusionBottleneck(nn.Module):
    """
    attention fusion on bottleneck feature (H/4, W/4)
    Add a small 3x3 context conv before producing weights (more stable than pointwise-only).
    """
    def __init__(self, c=128, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.context = nn.Sequential(
            ConvBNReLU(c * num_modalities, 128, k=3, s=1, p=1),
            ConvBNReLU(128, 64, k=3, s=1, p=1),
        )
        self.to_att = nn.Sequential(
            nn.Conv2d(64, num_modalities, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, feats_list):
        # feats_list: list of [B,C,H,W]
        stack = torch.stack(feats_list, dim=1)  # [B,M,C,H,W]
        B, M, C, H, W = stack.shape
        x = stack.view(B, M * C, H, W)
        ctx = self.context(x)
        att = self.to_att(ctx).unsqueeze(2)     # [B,M,1,H,W]
        fused = (att * stack).sum(dim=1)        # [B,C,H,W]
        return fused, att.squeeze(2)            # return att maps for debugging if needed


class DecoderUNetLite(nn.Module):
    def __init__(self, c1=32, c2=64, c3=128, out_channels=1):
        super().__init__()
        # up from /4 -> /2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c3, c2)
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(c2 + c2, c2),
            ConvBNReLU(c2, c2),
        )
        # up from /2 -> /1
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c2, c1)
        )
        self.dec1 = nn.Sequential(
            ConvBNReLU(c1 + c1, c1),
            ConvBNReLU(c1, c1),
        )
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, f1, f2, f3_fused):
        x = self.up1(f3_fused)                 # [B,c2,H/2,W/2]
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)
        x = self.up2(x)                        # [B,c1,H,W]
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)
        y = torch.sigmoid(self.head(x))        # [B,1,H,W]
        return y


class EdgeNetV2(nn.Module):
    """
    mode:
      - "multi": expects x_dict with all modalities; returns y_multi
      - "single": expects x_dict has only one modality key (or you pass single_mod),
                  returns y_single computed from that modality encoder only
    """
    def __init__(self, modalities=("natural", "infrared", "sar")):
        super().__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict({
            "natural":  EncoderUNetLite(3),
            "infrared": EncoderUNetLite(3),
            "sar":      EncoderUNetLite(3),
        })
        self.fusion = AttentionFusionBottleneck(c=128, num_modalities=len(modalities))
        self.decoder = DecoderUNetLite(out_channels=1)

        # optional: learnable temperature / strength (can help avoid saturation)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_dict, mode="multi", single_mod=None):
        if mode == "single":
            if single_mod is None:
                # infer which key exists
                keys = [k for k in x_dict.keys() if k in self.encoders]
                if len(keys) != 1:
                    raise ValueError(f"single mode expects exactly one modality, got keys={keys}")
                single_mod = keys[0]
            f1, f2, f3 = self.encoders[single_mod](x_dict[single_mod])
            y = self.decoder(f1, f2, f3)
            return y

        # multi mode
        f1_list, f2_list, f3_list = [], [], []
        for m in self.modalities:
            f1, f2, f3 = self.encoders[m](x_dict[m])
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)

        # shallow skip fusion: average (simple & stable). You can replace with attention if needed.
        f1_fused = torch.stack(f1_list, dim=0).mean(dim=0)
        f2_fused = torch.stack(f2_list, dim=0).mean(dim=0)

        f3_fused, att = self.fusion(f3_list)
        y = self.decoder(f1_fused, f2_fused, f3_fused)

        # optional logit scaling
        if self.logit_scale is not None:
            # y in (0,1); keep it unchanged for simplicity
            pass

        return y


# =========================================================
# Structure priors (optional, modality-fair)
# =========================================================
def sobel_mag_tensor(x):
    """
    x: [B,3,H,W] in [0,1]
    return: [B,1,H,W] gradient magnitude (normalized)
    """
    # convert to gray
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    # fixed Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=torch.float32, device=x.device).view(1,1,3,3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

    # normalize per-image to [0,1] roughly
    B = mag.shape[0]
    mag_flat = mag.view(B, -1)
    denom = (mag_flat.max(dim=1)[0].view(B,1,1,1) + 1e-6)
    mag = mag / denom
    return mag


def median_aggregate(a, b, c):
    # elementwise median of 3 tensors
    stack = torch.stack([a, b, c], dim=0)
    return stack.median(dim=0).values


def tv_loss(y):
    # y: [B,1,H,W]
    return (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean() + (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean()


def corr_loss(y, g):
    """
    1 - Pearson correlation (maximize correlation => minimize this)
    y,g: [B,1,H,W]
    """
    B = y.shape[0]
    yv = y.view(B, -1)
    gv = g.view(B, -1)
    yv = yv - yv.mean(dim=1, keepdim=True)
    gv = gv - gv.mean(dim=1, keepdim=True)
    num = (yv * gv).mean(dim=1)
    den = (yv.std(dim=1) * gv.std(dim=1) + 1e-6)
    corr = num / den
    return (1.0 - corr).mean()


# =========================================================
# Visualization (same style as your original)
# =========================================================
def visualize_modalities_with_all_preds(batch, preds_dict, epoch, out_dir="edgeY/val_outputs"):
    """
    batch: dict with keys natural/infrared/sar tensors on CPU (as dataloader gives)
    preds_dict: {
        "multi": [B,1,H,W],
        "natural": [B,1,H,W],
        "infrared": [B,1,H,W],
        "sar": [B,1,H,W],
    }
    """
    os.makedirs(f"{out_dir}/epoch_{epoch}", exist_ok=True)

    modalities = ["natural", "infrared", "sar"]
    pred_keys = ["multi", "natural", "infrared", "sar"]
    row_names = modalities + [f"y_{k}" for k in pred_keys]

    # 3 input rows + 4 output rows = 7 rows
    n_rows = 3 + 4
    n_cols = 4  # show first 4 samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

    # --- input images ---
    for col in range(n_cols):
        for r, mod in enumerate(modalities):
            img = batch[mod][col].permute(1, 2, 0).cpu().numpy()
            axes[r][col].imshow(img)
            axes[r][col].axis("off")
            if col == 0:
                axes[r][col].set_ylabel(mod, fontsize=12)

    # --- outputs ---
    for col in range(n_cols):
        for i, k in enumerate(pred_keys):
            r = 3 + i
            out = preds_dict[k][col, 0].detach().cpu().numpy()
            axes[r][col].imshow(out, cmap="gray")
            axes[r][col].axis("off")
            if col == 0:
                axes[r][col].set_ylabel(f"y_{k}", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/epoch_{epoch}/grid_all_preds.png")
    plt.close()

def to_uint8_autocontrast_minmax(img2d: np.ndarray) -> np.ndarray:
    """
    Mimic matplotlib imshow default autoscale (per-image min/max).
    img2d: float array, any range
    """
    vmin = float(np.min(img2d))
    vmax = float(np.max(img2d))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(img2d, dtype=np.uint8)
    out = (img2d - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

def save_all_val_outputs_all_modes(model, val_loader, device, out_dir="edgeY/val_outputs/final"):
    """
    Saves:
      out_dir/multi/*.png
      out_dir/natural/*.png
      out_dir/infrared/*.png
      out_dir/vh/*.png
    """
    model.eval()
    subdirs = ["multi", "natural", "infrared", "vh"]
    for sd in subdirs:
        os.makedirs(os.path.join(out_dir, sd), exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving full val set (multi + single)"):
            x = {k: v.to(device) for k, v in batch.items() if k != "filename"}
            filenames = batch["filename"]

            # --- forward ---
            y_multi = model(x, mode="multi")
            y_nat  = model({"natural": x["natural"]}, mode="single", single_mod="natural")
            y_ir   = model({"infrared": x["infrared"]}, mode="single", single_mod="infrared")
            y_sar  = model({"sar": x["sar"]}, mode="single", single_mod="sar")

            preds = {
                "multi": y_multi,
                "natural": y_nat,
                "infrared": y_ir,
                "vh": y_sar
            }

            # --- save per sample ---
            for i in range(y_multi.shape[0]):
                fname = filenames[i]
                for k, y in preds.items():
                    arr = y[i, 0].cpu().numpy() 
                    y_np = to_uint8_autocontrast_minmax(arr)
                    cv2.imwrite(os.path.join(out_dir, k, k+"_"+fname), y_np)




# =========================================================
# Training
# =========================================================
def train(model, train_loader, val_loader, device, epochs=50,out_dir="edgeY/val_outputs"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- loss weights ----
    # 1) density constraint: keep y neither too dark nor too bright
    rho = 0.10
    w_density = 1.0

    # 2) multi-teacher -> single-student distillation (key for single-modality inference)
    w_distill = 10.0

    # 3) optional TV to suppress noisy speckles without pushing mean to 0
    w_tv = 0.05

    # 4) optional modality-fair structure prior (does NOT rely on visible pseudo edge)
    use_structure_prior = True
    w_prior = 0.5

    best = 1e9
    seed_all(42)

    # sample fixed val batch for visualization
    val_batches = list(val_loader)
    sampled_val_batch = random.sample(val_batches, 1)[0]  # assumes batch_size >= 4

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x = {k: v.to(device) for k, v in batch.items() if k != "filename"}

            # ---- forward ----
            y_multi = model(x, mode="multi")

            y_nat = model({"natural": x["natural"]}, mode="single", single_mod="natural")
            y_ir  = model({"infrared": x["infrared"]}, mode="single", single_mod="infrared")
            y_sar = model({"sar": x["sar"]}, mode="single", single_mod="sar")

            # ---- losses ----
            # (A) density constraint (replace y.mean())
            loss_density = (y_multi.mean() - rho).pow(2)

            # (B) distillation: single outputs learn to match fused teacher
            teacher = y_multi.detach()
            loss_distill = (F.l1_loss(y_nat, teacher) +
                            F.l1_loss(y_ir, teacher) +
                            F.l1_loss(y_sar, teacher))

            # (C) TV (optional)
            loss_tv = tv_loss(y_multi)

            # (D) structure prior (optional): align y with median gradient magnitude
            loss_prior = torch.tensor(0.0, device=device)
            if use_structure_prior:
                # SAR often benefits from log compression before gradients; you can customize here
                g_nat = sobel_mag_tensor(x["natural"])
                g_ir  = sobel_mag_tensor(x["infrared"])
                g_sar = sobel_mag_tensor(x["sar"])
                g_agg = median_aggregate(g_nat, g_ir, g_sar)
                loss_prior = corr_loss(y_multi, g_agg)

            loss = (w_density * loss_density +
                    w_distill * loss_distill +
                    w_tv * loss_tv +
                    w_prior * loss_prior)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(train_loader)
        print(f"Epoch {epoch} | TrainLoss={avg:.4f} | density={loss_density.item():.4f} "
              f"| distill={loss_distill.item():.4f} | tv={loss_tv.item():.4f} | prior={loss_prior.item():.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), out_dir+f"/weight/model_best.pth")

        if (epoch - 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                xb = {k: v.to(device) for k, v in sampled_val_batch.items() if k != "filename"}

                y_multi = model(xb, mode="multi")
                y_nat  = model({"natural": xb["natural"]}, mode="single", single_mod="natural")
                y_ir   = model({"infrared": xb["infrared"]}, mode="single", single_mod="infrared")
                y_sar  = model({"sar": xb["sar"]}, mode="single", single_mod="sar")

                preds_dict = {
                    "multi": y_multi,
                    "natural": y_nat,
                    "infrared": y_ir,
                    "sar": y_sar,
                }

                print("Val stats:",
                      "multi(mean/min/max)=", y_multi.mean().item(), y_multi.min().item(), y_multi.max().item(),
                      "| nat(mean)=", y_nat.mean().item(),
                      "| ir(mean)=", y_ir.mean().item(),
                      "| sar(mean)=", y_sar.mean().item())

                # 注意：这里 batch 是 CPU 的 sampled_val_batch，用于展示输入图；preds 在 GPU/CPU 都行
                visualize_modalities_with_all_preds(sampled_val_batch, preds_dict, epoch, out_dir=out_dir+f"valoutput/epoch_{epoch}")

            torch.save(model.state_dict(), out_dir+f"/weight/model_epoch{epoch}.pth")
    save_all_val_outputs_all_modes(model, val_loader, device, out_dir=out_dir+f"valoutput/final")

def generate_from_best(val_loader, device, ckpt_path="model_best.pth", out_dir=None):
    model = EdgeNetV2(modalities=("natural", "infrared", "sar")).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    print(f"[RUN] Saving outputs to: {out_dir}")

    save_all_val_outputs_all_modes(model, val_loader, device, out_dir=out_dir)

    print("[DONE] All outputs saved.")
# =========================================================
# Entry
# =========================================================
# if __name__ == "__main__":
#     train_loader = get_loader(
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s2/natural",
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s2/infrared",
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s1/vh",
#         batch_size=4,
#         shuffle=True,
#         num_workers=2
#     )

#     val_loader = get_loader(
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural",
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/infrared",
#         "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh",
#         batch_size=4,
#         shuffle=False,
#         num_workers=2
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = EdgeNetV2(modalities=("natural", "infrared", "sar"))
#     train(model, train_loader, val_loader, device, epochs=50,out_dir="edgeY/log")

if __name__ == "__main__":
    # 只需要 val_loader
    val_loader = get_loader(
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/infrared",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh",
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成 multi + 单模态的最终结果
    generate_from_best(
        val_loader=val_loader,
        device=device,
        ckpt_path="edgeY/weight/model_best.pth",          # best 权重路径（同目录）
        out_dir="edgeY/val_outputs"     # 输出目录（避免覆盖）
    )

