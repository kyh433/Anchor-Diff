import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def pil_to_numpy(images):
    if not isinstance(images, list):
        images = [images]
    # 归一化到 [0, 1]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)
    return images

def process_multimodal_data():
    stats = ['train', 'val', 'test']
    
    # 定义模态映射：(原始路径名, 边缘路径名, 传感器类型)
    modalities = [
        ("natural", "natural", "s2"),
        ("infrared", "infrared", "s2"),
        ("vh", "vh", "s1")
    ]

    base_path = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset"
    edge_base = "/mnt/data/project_kyh/MultimodalityGeneration25/edgeY"

    for stat in stats:
        print(f"正在处理数据集类型: {stat}")
        
        for mod_name, edge_name, sensor in modalities:
            # 构建输入、边缘和输出路径
            img_dir = os.path.join(base_path, stat, sensor, mod_name)
            edge_dir = os.path.join(edge_base, f"{stat}_outputs", edge_name)
            save_dir = os.path.join(base_path, stat, sensor, f"{mod_name}_fusion_edge")
            
            os.makedirs(save_dir, exist_ok=True)

            # 获取所有图片文件名
            if not os.path.exists(img_dir):
                continue
            
            filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

            for fname in tqdm(filenames, desc=f"Processing {mod_name}"):
                try:
                    # 1. 读取原始图像并转为 numpy (B, H, W, C)
                    img_path = os.path.join(img_dir, fname)
                    raw_img = Image.open(img_path).convert('RGB')
                    img_np = pil_to_numpy(raw_img)[0]  # 取出第一张，形状为 (H, W, 3)

                    # 2. 读取边缘图像
                    edge_path = os.path.join(edge_dir, fname)
                    if not os.path.exists(edge_path):
                        print(f"警告: 边缘文件缺失 {edge_path}")
                        continue
                    
                    # 取边缘图的一个通道并归一化
                    edge_img = Image.open(edge_path).convert('L') # 直接转灰度
                    edge_np = np.array(edge_img).astype(np.float32) / 255.0
                    edge_np = np.expand_dims(edge_np, axis=-1)  # 形状变为 (H, W, 1)

                    # 3. 在第四个通道叠加 (Concatenate)
                    # 结果形状: (H, W, 4)
                    fusion_np = np.concatenate([img_np, edge_np], axis=-1)
                    fusion_np = np.transpose(fusion_np, (2, 0, 1))
                    # 4. 保存为 .npy 格式
                    save_name = os.path.splitext(fname)[0] + ".npy"
                    np.save(os.path.join(save_dir, save_name), fusion_np)
                    
                except Exception as e:
                    print(f"处理文件 {fname} 时出错: {e}")

if __name__ == "__main__":
    process_multimodal_data()