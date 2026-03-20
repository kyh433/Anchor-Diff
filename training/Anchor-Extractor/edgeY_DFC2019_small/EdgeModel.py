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
# Dataset
# =========================================================
def get_preprocessing_transform(size=(512, 512)):
    """Percentile contrast stretching per channel + resize + tensor."""
    def stretch(img):
        img_np = np.array(img).astype(np.float32)
        for i in range(3):
            p2 = np.percentile(img_np[..., i], 2)
            p98 = np.percentile(img_np[..., i], 98)
            if p98 > p2:
                img_np[..., i] = (img_np[..., i] - p2) / (p98 - p2)
            else:
                img_np[..., i] = 0
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    return transforms.Compose([
        transforms.Lambda(stretch),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])


class MultiModalDataset(Dataset):
    """
    Match files like:
      JAX_004_006_MSI_0_0.png
      JAX_004_006_RGB_0_0.png
      JAX_004_006_AGL_0_0.png

    Pair key becomes:
      JAX_004_006_0_0
    """
    def __init__(self, rgb_dir, msi_dir, agl_dir, transform=None):
        self.transform = transform

        rgb_paths = glob.glob(os.path.join(rgb_dir, "*.png"))
        msi_paths = glob.glob(os.path.join(msi_dir, "*.png"))
        agl_paths = glob.glob(os.path.join(agl_dir, "*.png"))

        # Example: JAX_004_006_MSI_0_0.png
        pat = re.compile(r"^(.*)_(RGB|MSI|AGL)_(.*)\.png$", re.IGNORECASE)

        def build_id_map(paths, expected_mod):
            mapping = {}
            expected_mod = expected_mod.lower()
            for p in paths:
                name = os.path.basename(p)
                mm = pat.match(name)
                if mm is None:
                    continue

                prefix = mm.group(1)
                mod = mm.group(2).lower()
                suffix = mm.group(3)

                if mod != expected_mod:
                    continue

                pair_id = f"{prefix}_{suffix}"
                mapping[pair_id] = p
            return mapping

        rgb_map = build_id_map(rgb_paths, "rgb")
        msi_map = build_id_map(msi_paths, "msi")
        agl_map = build_id_map(agl_paths, "agl")

        keys = sorted(list(set(rgb_map.keys()) & set(msi_map.keys()) & set(agl_map.keys())))
        if len(keys) == 0:
            rgb_any = next(iter(rgb_map.keys()), None)
            msi_any = next(iter(msi_map.keys()), None)
            agl_any = next(iter(agl_map.keys()), None)
            raise RuntimeError(
                "No matched IDs across RGB/MSI/AGL folders.\n"
                f"Parsed counts: RGB={len(rgb_map)}, MSI={len(msi_map)}, AGL={len(agl_map)}\n"
                f"Example IDs: RGB={rgb_any}, MSI={msi_any}, AGL={agl_any}\n"
                "Check filename pattern like 'JAX_004_006_MSI_0_0.png'."
            )

        self.samples = [(rgb_map[k], msi_map[k], agl_map[k], k) for k in keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, msi_path, agl_path, pair_id = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        msi = Image.open(msi_path).convert("RGB")
        agl = Image.open(agl_path).convert("RGB")

        if self.transform:
            rgb = self.transform(rgb)
            msi = self.transform(msi)
            agl = self.transform(agl)

        return {"rgb": rgb, "msi": msi, "agl": agl, "filename": f"{pair_id}.png"}



def get_loader(rgb_dir, msi_dir, agl_dir, batch_size=4, shuffle=True, num_workers=2):
    tfm = get_preprocessing_transform()
    ds = MultiModalDataset(rgb_dir, msi_dir, agl_dir, tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# =========================================================
# Model
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class EncoderUNetLite(nn.Module):
    """
    returns:
      f1: [B, C1, H,   W  ]
      f2: [B, C2, H/2, W/2]
      f3: [B, C3, H/4, W/4]
    """
    def __init__(self, in_channels=3, c1=32, c2=64, c3=128):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, c1),
            ConvBNReLU(c1, c1),
        )
        self.down1 = ConvBNReLU(c1, c2, s=2)
        self.stage2 = nn.Sequential(
            ConvBNReLU(c2, c2),
            ConvBNReLU(c2, c2),
        )
        self.down2 = ConvBNReLU(c2, c3, s=2)
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
        stack = torch.stack(feats_list, dim=1)  # [B,M,C,H,W]
        B, M, C, H, W = stack.shape
        x = stack.view(B, M * C, H, W)
        ctx = self.context(x)
        att = self.to_att(ctx).unsqueeze(2)     # [B,M,1,H,W]
        fused = (att * stack).sum(dim=1)        # [B,C,H,W]
        return fused, att.squeeze(2)


class DecoderUNetLite(nn.Module):
    def __init__(self, c1=32, c2=64, c3=128, out_channels=1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c3, c2)
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(c2 + c2, c2),
            ConvBNReLU(c2, c2),
        )
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
        x = self.up1(f3_fused)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)
        y = torch.sigmoid(self.head(x))
        return y


class EdgeNetV2(nn.Module):
    """
    mode:
      - "multi": expects x_dict with all modalities; returns y_multi
      - "single": expects x_dict has one modality key; returns y_single
    """
    def __init__(self, modalities=("rgb", "msi", "agl")):
        super().__init__()
        self.modalities = tuple(modalities)
        self.encoders = nn.ModuleDict({
            "rgb": EncoderUNetLite(3),
            "msi": EncoderUNetLite(3),
            "agl": EncoderUNetLite(3),
        })
        self.fusion = AttentionFusionBottleneck(c=128, num_modalities=len(self.modalities))
        self.decoder = DecoderUNetLite(out_channels=1)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_dict, mode="multi", single_mod=None):
        if mode == "single":
            if single_mod is None:
                keys = [k for k in x_dict.keys() if k in self.encoders]
                if len(keys) != 1:
                    raise ValueError(f"single mode expects exactly one modality, got keys={keys}")
                single_mod = keys[0]
            f1, f2, f3 = self.encoders[single_mod](x_dict[single_mod])
            y = self.decoder(f1, f2, f3)
            return y

        f1_list, f2_list, f3_list = [], [], []
        for m in self.modalities:
            if m not in x_dict:
                raise KeyError(f"Missing modality '{m}' in x_dict. Available keys: {list(x_dict.keys())}")
            f1, f2, f3 = self.encoders[m](x_dict[m])
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)

        f1_fused = torch.stack(f1_list, dim=0).mean(dim=0)
        f2_fused = torch.stack(f2_list, dim=0).mean(dim=0)
        f3_fused, _ = self.fusion(f3_list)
        y = self.decoder(f1_fused, f2_fused, f3_fused)
        return y


# =========================================================
# Structure priors
# =========================================================
def sobel_mag_tensor(x):
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

    B = mag.shape[0]
    mag_flat = mag.view(B, -1)
    denom = mag_flat.max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
    mag = mag / denom
    return mag



def median_aggregate(a, b, c):
    stack = torch.stack([a, b, c], dim=0)
    return stack.median(dim=0).values



def tv_loss(y):
    return (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean() + (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean()



def corr_loss(y, g):
    B = y.shape[0]
    yv = y.view(B, -1)
    gv = g.view(B, -1)
    yv = yv - yv.mean(dim=1, keepdim=True)
    gv = gv - gv.mean(dim=1, keepdim=True)
    num = (yv * gv).mean(dim=1)
    den = yv.std(dim=1) * gv.std(dim=1) + 1e-6
    corr = num / den
    return (1.0 - corr).mean()


# =========================================================
# Visualization
# =========================================================
def visualize_modalities_with_all_preds(batch, preds_dict, epoch, out_dir="edgeY_DFC2019_small/val_outputs"):
    epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    modalities = ["rgb", "msi", "agl"]
    pred_keys = ["multi", "rgb", "msi", "agl"]

    n_rows = 3 + 4
    batch_n = batch[modalities[0]].shape[0]
    n_cols = min(4, batch_n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(n_cols):
        for r, mod in enumerate(modalities):
            img = batch[mod][col].permute(1, 2, 0).cpu().numpy()
            axes[r][col].imshow(img)
            axes[r][col].axis("off")
            if col == 0:
                axes[r][col].set_ylabel(mod, fontsize=12)

    for col in range(n_cols):
        for i, k in enumerate(pred_keys):
            r = 3 + i
            out = preds_dict[k][col, 0].detach().cpu().numpy()
            axes[r][col].imshow(out, cmap="gray")
            axes[r][col].axis("off")
            if col == 0:
                axes[r][col].set_ylabel(f"y_{k}", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, "grid_all_preds.png"))
    plt.close()



def to_uint8_autocontrast_minmax(img2d: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img2d))
    vmax = float(np.max(img2d))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(img2d, dtype=np.uint8)
    out = (img2d - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)



def save_all_val_outputs_all_modes(model, val_loader, device, out_dir="edgeY_DFC2019_small/val_outputs/final"):
    model.eval()
    subdirs = ["multi", "rgb", "msi", "agl"]
    for sd in subdirs:
        os.makedirs(os.path.join(out_dir, sd), exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving full val set (multi + single)"):
            x = {k: v.to(device) for k, v in batch.items() if k != "filename"}
            filenames = batch["filename"]

            y_multi = model(x, mode="multi")
            y_rgb = model({"rgb": x["rgb"]}, mode="single", single_mod="rgb")
            y_msi = model({"msi": x["msi"]}, mode="single", single_mod="msi")
            y_agl = model({"agl": x["agl"]}, mode="single", single_mod="agl")

            preds = {
                "multi": y_multi,
                "rgb": y_rgb,
                "msi": y_msi,
                "agl": y_agl,
            }

            for i in range(y_multi.shape[0]):
                fname = filenames[i]
                for k, y in preds.items():
                    arr = y[i, 0].cpu().numpy()
                    y_np = to_uint8_autocontrast_minmax(arr)
                    cv2.imwrite(os.path.join(out_dir, k, f"{k}_{fname}"), y_np)


# =========================================================
# Training
# =========================================================
def train(model, train_loader, val_loader, device, epochs=50, out_dir="edgeY_DFC2019_small/log"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "weight"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "valoutput"), exist_ok=True)

    rho = 0.10
    w_density = 1.0
    w_distill = 10.0
    w_tv = 0.05
    use_structure_prior = True
    w_prior = 0.5

    best = 1e9
    seed_all(42)

    sampled_val_batch = next(iter(val_loader))

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x = {k: v.to(device) for k, v in batch.items() if k != "filename"}

            y_multi = model(x, mode="multi")
            y_rgb = model({"rgb": x["rgb"]}, mode="single", single_mod="rgb")
            y_msi = model({"msi": x["msi"]}, mode="single", single_mod="msi")
            y_agl = model({"agl": x["agl"]}, mode="single", single_mod="agl")

            loss_density = (y_multi.mean() - rho).pow(2)
            teacher = y_multi.detach()
            loss_distill = (
                F.l1_loss(y_rgb, teacher) +
                F.l1_loss(y_msi, teacher) +
                F.l1_loss(y_agl, teacher)
            )
            loss_tv = tv_loss(y_multi)

            loss_prior = torch.tensor(0.0, device=device)
            if use_structure_prior:
                g_rgb = sobel_mag_tensor(x["rgb"])
                g_msi = sobel_mag_tensor(x["msi"])
                g_agl = sobel_mag_tensor(x["agl"])
                g_agg = median_aggregate(g_rgb, g_msi, g_agl)
                loss_prior = corr_loss(y_multi, g_agg)

            loss = (
                w_density * loss_density +
                w_distill * loss_distill +
                w_tv * loss_tv +
                w_prior * loss_prior
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / max(len(train_loader), 1)
        print(
            f"Epoch {epoch} | TrainLoss={avg:.4f} | density={loss_density.item():.4f} "
            f"| distill={loss_distill.item():.4f} | tv={loss_tv.item():.4f} | prior={loss_prior.item():.4f}"
        )

        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out_dir, "weight", "model_best.pth"))

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                xb = {k: v.to(device) for k, v in sampled_val_batch.items() if k != "filename"}

                y_multi = model(xb, mode="multi")
                y_rgb = model({"rgb": xb["rgb"]}, mode="single", single_mod="rgb")
                y_msi = model({"msi": xb["msi"]}, mode="single", single_mod="msi")
                y_agl = model({"agl": xb["agl"]}, mode="single", single_mod="agl")

                preds_dict = {
                    "multi": y_multi,
                    "rgb": y_rgb,
                    "msi": y_msi,
                    "agl": y_agl,
                }

                print(
                    "Val stats:",
                    "multi(mean/min/max)=", y_multi.mean().item(), y_multi.min().item(), y_multi.max().item(),
                    "| rgb(mean)=", y_rgb.mean().item(),
                    "| msi(mean)=", y_msi.mean().item(),
                    "| agl(mean)=", y_agl.mean().item(),
                )

                visualize_modalities_with_all_preds(
                    sampled_val_batch,
                    preds_dict,
                    epoch,
                    out_dir=os.path.join(out_dir, "valoutput")
                )

            torch.save(model.state_dict(), os.path.join(out_dir, "weight", f"model_epoch{epoch}.pth"))

    save_all_val_outputs_all_modes(model, val_loader, device, out_dir=os.path.join(out_dir, "valoutput", "final"))



def generate_from_best(val_loader, device, ckpt_path="model_best.pth", out_dir=None):
    model = EdgeNetV2(modalities=("rgb", "msi", "agl")).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if out_dir is None:
        out_dir = "edgeY_DFC2019_small/val_outputs_from_best"

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
if __name__ == "__main__":
    train_loader = get_loader(
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/Track1-RGB",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/Track1-MSI",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/Track1-AGL",
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    val_loader = get_loader(
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/test/Track1-RGB",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/test/Track1-MSI",
        "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/processed_DFC2019_small/test/Track1-AGL",
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeNetV2(modalities=("rgb", "msi", "agl"))
    train(model, train_loader, val_loader, device, epochs=50, out_dir="edgeY_DFC2019_small/log")
