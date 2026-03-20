import os
import numpy as np
from PIL import Image

# ========== 基础工具 ==========

def load_images_from_folder(folder_path):
    """
    按文件名排序，读取文件夹下所有 PNG，返回 PIL.Image 列表和文件名列表
    """
    images = []
    filenames = []
    image_files = sorted(os.listdir(folder_path))
    for image_file in image_files:
        if image_file.lower().endswith(".png"):
            img_path = os.path.join(folder_path, image_file)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            filenames.append(image_file)
    return images, filenames


def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    """
    PIL RGB/其他模式 -> 灰度 np.ndarray, float32, 范围 [0,1]
    """
    gray = img.convert("L")  # (H, W), uint8 [0,255]
    arr = np.array(gray, dtype=np.float32) / 255.0
    return arr


# ========== 卷积 + 结构特征提取（梯度） ==========

def conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    简单 2D 卷积 (same padding, reflect 边界)
    img: 2D array
    kernel: 2D array
    """
    assert img.ndim == 2, "conv2d_same expects 2D array"
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    # 只在 kernel 维度上做小循环，仍然比较快
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + img.shape[0], j : j + img.shape[1]]
    return out


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """
    用 Sobel 算子计算梯度幅值，返回归一化到 [0,1] 的结构图
    """
    gray = gray.astype(np.float32)
    # Sobel kernel
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = conv2d_same(gray, Kx)
    gy = conv2d_same(gray, Ky)
    mag = np.sqrt(gx * gx + gy * gy)
    # 归一化到 [0,1]
    m_min, m_max = mag.min(), mag.max()
    if m_max > m_min:
        mag = (mag - m_min) / (m_max - m_min)
    else:
        mag = np.zeros_like(mag)
    return mag.astype(np.float32)


# ========== NMI 核心 ==========

def calculate_nmi(img_a: np.ndarray,
                  img_b: np.ndarray,
                  bins: int = 64,
                  eps: float = 1e-10,
                  variant: str = "A",
                  normalize: bool = True) -> float:
    """
    img_a, img_b: 2D numpy arrays (同尺寸)
    bins: 直方图 bins
    variant:
        "A": NMI = (H(A)+H(B))/H(A,B)   (值通常 >=1, 独立时 ~1)
        "B": NMI = 2*MI/(H(A)+H(B))     (大致在 [0,1])
    normalize:
        True: 把各自归一化到 [0,1] 并在该范围上统计 joint hist
    """
    assert img_a.shape == img_b.shape, "Images must have same shape."
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    if normalize:
        a_min, a_max = a.min(), a.max()
        b_min, b_max = b.min(), b.max()
        if a_max > a_min:
            a = (a - a_min) / (a_max - a_min)
        else:
            a = np.zeros_like(a)
        if b_max > b_min:
            b = (b - b_min) / (b_max - b_min)
        else:
            b = np.zeros_like(b)
        hist_range = [[0.0, 1.0], [0.0, 1.0]]
    else:
        hist_range = None

    a_flat = a.ravel()
    b_flat = b.ravel()

    joint_hist, _, _ = np.histogram2d(
        a_flat, b_flat, bins=bins, range=hist_range
    )
    joint_prob = joint_hist / (np.sum(joint_hist) + eps)

    # 边缘分布
    p_a = np.sum(joint_prob, axis=1)
    p_b = np.sum(joint_prob, axis=0)

    # 熵
    H_a = -np.sum(p_a * np.log(p_a + eps))
    H_b = -np.sum(p_b * np.log(p_b + eps))
    H_ab = -np.sum(joint_prob * np.log(joint_prob + eps))

    MI = H_a + H_b - H_ab

    if variant == "A":
        return float((H_a + H_b) / (H_ab + eps))
    else:
        return float(2.0 * MI / (H_a + H_b + eps))


# ========== 基于结构特征的全局 NMI ==========

def structural_nmi(gray_a: np.ndarray,
                   gray_b: np.ndarray,
                   bins: int = 64,
                   variant: str = "B") -> float:
    """
    先把灰度图转换成结构图（梯度幅值），再计算全图 NMI
    """
    feat_a = compute_gradient_magnitude(gray_a)
    feat_b = compute_gradient_magnitude(gray_b)
    nmi_val = calculate_nmi(feat_a, feat_b, bins=bins, variant=variant, normalize=True)
    return nmi_val


# ========== patch-based 局部 NMI ==========

def patchwise_nmi(gray_a: np.ndarray,
                  gray_b: np.ndarray,
                  patch_size: int = 32,
                  stride: int = 32,
                  bins: int = 32,
                  variant: str = "B",
                  min_std: float = 1e-3):
    """
    在结构特征图上做 patch-based NMI:
      1. 灰度 -> 梯度结构图
      2. 在结构图上滑动窗口计算 NMI
      3. 返回所有 patch NMI 的均值 / 方差 / 以及 NMI map (可选)

    返回:
      mean_nmi: 所有 patch NMI 的均值
      std_nmi : 所有 patch NMI 的标准差
      nmi_map : 粗分辨率 NMI 图 (grid_h x grid_w)，可视化用；不需要可以忽略
    """
    assert gray_a.shape == gray_b.shape, "Images must have same shape."

    feat_a = compute_gradient_magnitude(gray_a)
    feat_b = compute_gradient_magnitude(gray_b)

    H, W = feat_a.shape
    ps = patch_size
    st = stride

    values = []
    centers = []

    for y in range(0, H - ps + 1, st):
        for x in range(0, W - ps + 1, st):
            pa = feat_a[y : y + ps, x : x + ps]
            pb = feat_b[y : y + ps, x : x + ps]

            # 跳过几乎全常数的 patch，避免大面积背景/海洋等区域干扰
            if pa.std() < min_std and pb.std() < min_std:
                continue

            v = calculate_nmi(pa, pb, bins=bins, variant=variant, normalize=True)
            values.append(v)
            centers.append((y + ps // 2, x + ps // 2))

    if len(values) == 0:
        # 所有 patch 都被跳过，返回 NaN
        return np.nan, np.nan, None

    values = np.array(values, dtype=np.float32)
    mean_nmi = float(values.mean())
    std_nmi = float(values.std())

    # 可选：生成一个粗分辨率的 NMI map（以 patch 网格为尺度）
    grid_h = (H - ps) // st + 1
    grid_w = (W - ps) // st + 1
    nmi_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

    idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            y = gy * st
            x = gx * st
            if y <= H - ps and x <= W - ps:
                if idx < len(values):
                    nmi_map[gy, gx] = values[idx]
                    idx += 1

    return mean_nmi, std_nmi, nmi_map


# ========== 封装成“两个文件夹之间的结构+局部 NMI 评估” ==========

def compute_structural_and_patch_nmi_for_folders(
    folder_a: str,
    folder_b: str,
    patch_size: int = 32,
    stride: int = 32,
    bins_global: int = 64,
    bins_patch: int = 32,
    variant_global: str = "B",
    variant_patch: str = "B",
):
    """
    folder_a: 例如 光学/参考模态
    folder_b: 例如 SAR / 生成模态
    输出：
      - 全局结构 NMI 的均值/方差
      - patch-based NMI 的均值/方差
    """
    imgs_a, names_a = load_images_from_folder(folder_a)
    imgs_b, names_b = load_images_from_folder(folder_b)

    n = min(len(imgs_a), len(imgs_b))
    if len(imgs_a) != len(imgs_b):
        print(f"[Warn] folder sizes differ: {len(imgs_a)} vs {len(imgs_b)}, will use first {n} pairs.")

    global_nmis = []
    patch_nmis = []

    for i in range(n):
        name_a = names_a[i]
        name_b = names_b[i]
        img_a = imgs_a[i]
        img_b = imgs_b[i]

        gray_a = pil_to_gray_np(img_a)
        gray_b = pil_to_gray_np(img_b)

        # 保证同尺寸（如果不一样，这里可以改成 resize；现在先强制一致）
        assert gray_a.shape == gray_b.shape, f"shape mismatch for pair {name_a} vs {name_b}: {gray_a.shape} vs {gray_b.shape}"

        # 全局结构 NMI
        g_nmi = structural_nmi(gray_a, gray_b, bins=bins_global, variant=variant_global)
        # 局部 patch NMI
        p_mean, p_std, _ = patchwise_nmi(
            gray_a, gray_b,
            patch_size=patch_size,
            stride=stride,
            bins=bins_patch,
            variant=variant_patch,
        )

        global_nmis.append(g_nmi)
        patch_nmis.append(p_mean)

        print(f"[{i+1:03d}/{n:03d}] {name_a} <-> {name_b} | "
              f"global_struct_NMI={g_nmi:.4f}, patch_mean_NMI={p_mean:.4f}, patch_std={p_std:.4f}")

    global_nmis = np.array(global_nmis, dtype=np.float32)
    patch_nmis = np.array(patch_nmis, dtype=np.float32)

    print("\n==== Summary (Structural Global NMI) ====")
    print(f"mean={global_nmis.mean():.4f}, std={global_nmis.std():.4f}")

    print("\n==== Summary (Patch-wise Structural NMI mean over images) ====")
    print(f"mean={patch_nmis.mean():.4f}, std={patch_nmis.std():.4f}")

    return {
        "global_struct_nmi_mean": float(global_nmis.mean()),
        "global_struct_nmi_std": float(global_nmis.std()),
        "patch_struct_nmi_mean": float(patch_nmis.mean()),
        "patch_struct_nmi_std": float(patch_nmis.std()),
    }


if __name__ == "__main__":
    # 示例：这里改成你自己的两个模态文件夹
    # 比如：folder_optical = "/mnt/.../natural"
    #      folder_sar     = "/mnt/.../vh"
    edit_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh"
    reconstruct_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural"

    edit_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/beta_output/natPCA2s1_vh_attention/edit"
    reconstruct_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/beta_output/natPCA2s1_vh_attention/res"

    folder_optical = "/path/to/optical"   # 可见光 / natural
    folder_sar     = "/path/to/sar"       # SAR / vh

    stats = compute_structural_and_patch_nmi_for_folders(
        edit_folder,
        reconstruct_folder,
        patch_size=32,
        stride=32,
        bins_global=32,
        bins_patch=32,
        variant_global="B",
        variant_patch="B",
    )
    # print("\nReturned stats:", stats)
