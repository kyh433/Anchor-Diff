import os
import numpy as np
from PIL import Image

# ====== load images (same style as your metrics code) ======
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    image_files = sorted(os.listdir(folder_path))
    for image_file in image_files:
        if image_file.endswith(".png"):
            img_path = os.path.join(folder_path, image_file)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            filenames.append(image_file)
    return images, filenames


# ====== NMI core ======
def calculate_nmi(img_a, img_b, bins=128, eps=1e-10, variant="A"):
    """
    img_a, img_b: 2D numpy arrays (grayscale), same shape
    bins: histogram bins
    variant:
        "A": NMI = (H(A)+H(B))/H(A,B)   (遥感配准常用，值通常 >=1)
        "B": NMI = 2*MI/(H(A)+H(B))    (值通常在 [0,1])
    """
    assert img_a.shape == img_b.shape, "Images must have same shape."

    a = img_a.ravel()
    b = img_b.ravel()

    # joint histogram -> joint probability
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    joint_prob = joint_hist / (np.sum(joint_hist) + eps)

    # marginals
    p_a = np.sum(joint_prob, axis=1)
    p_b = np.sum(joint_prob, axis=0)

    # entropies
    H_a  = -np.sum(p_a * np.log(p_a + eps))
    H_b  = -np.sum(p_b * np.log(p_b + eps))
    H_ab = -np.sum(joint_prob * np.log(joint_prob + eps))

    MI = H_a + H_b - H_ab

    if variant == "A":
        return (H_a + H_b) / (H_ab + eps)
    else:
        return 2.0 * MI / (H_a + H_b + eps)


# ====== main compute ======
def compute_nmi_for_folders(edit_folder, reconstruct_folder, bins=128, variant="A"):
    generated_images_edit, edit_names = load_images_from_folder(edit_folder)
    generated_images_reconstruct, recon_names = load_images_from_folder(reconstruct_folder)

    n = min(len(generated_images_edit), len(generated_images_reconstruct))
    if len(generated_images_edit) != len(generated_images_reconstruct):
        print(f"[Warn] folder sizes differ: edit={len(generated_images_edit)}, recon={len(generated_images_reconstruct)}. "
              f"Only first {n} pairs will be used.")

    nmi_scores = []
    for i, (edit_image, recon_image) in enumerate(zip(generated_images_edit[:n], generated_images_reconstruct[:n])):
        edit_gray  = np.array(edit_image.convert("L"), dtype=np.float32)
        recon_gray = np.array(recon_image.convert("L"), dtype=np.float32)

        score = calculate_nmi(edit_gray, recon_gray, bins=bins, variant=variant)
        nmi_scores.append(score)

        print(f"{i:04d} | {edit_names[i]}  <->  {recon_names[i]} | NMI: {score:.4f}")

    nmi_scores = np.array(nmi_scores)
    print("\n==== Summary ====")
    print(f"NMI mean: {nmi_scores.mean():.4f}")
    print(f"NMI std : {nmi_scores.std():.4f}")

    return float(nmi_scores.mean()), float(nmi_scores.std())


if __name__ == "__main__":
    # 改成你自己的两组生成结果路径
    edit_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh"
    reconstruct_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural"

    compute_nmi_for_folders(edit_folder, reconstruct_folder, bins=32, variant="B")
