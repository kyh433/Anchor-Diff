from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np
def auto_tone(image):
    """
    实现自动色调（Auto Tone）: 调整图像的色阶，使得图像的最亮和最暗部分映射到最大和最小的像素值。
    """
    img_array = np.array(image, dtype=np.float32)

    # 分别获取每个通道的最小值和最大值
    min_vals = np.min(img_array, axis=(0, 1))
    max_vals = np.max(img_array, axis=(0, 1))
    
    # 进行色阶调整，将每个通道的最小值映射为0，最大值映射为255
    img_array = (img_array - min_vals) / (max_vals - min_vals) * 255
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # 转换回PIL图像
    return Image.fromarray(img_array)
def divide_blend(image1, image2):
    """
    实现 Photoshop 中的 Divide 模式: 结果色 = (基色 / 混合色) * 255
    """
    # 将图像转换为 numpy 数组
    img1_array = np.array(image1, dtype=np.float32)
    img2_array = np.array(image2, dtype=np.float32)
    
    # 避免除零错误，替换零值为一个非常小的数值
    img2_array = np.where(img2_array == 0, 1, img2_array)
    
    # 执行 Divide（像素除法），然后乘以 255
    result_array = (img1_array / img2_array) * 255
    
    # 确保结果值在 0-255 之间
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    # 转换回 PIL 图像
    result_image = Image.fromarray(result_array)
    return result_image

def process_images(address1, address2, address3):
    # 获取地址1和地址2中的所有图像文件
    visible_images = [f for f in os.listdir(address1) if f.endswith(('.png', '.jpg', '.jpeg'))]
    edge_images = [f for f in os.listdir(address2) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 遍历可见光图像和边缘图像
    for visible_img, edge_img in zip(visible_images, edge_images):
        # 构建完整路径
        visible_path = os.path.join(address1, visible_img)
        edge_path = os.path.join(address2, edge_img)
        output_path = os.path.join(address3, visible_img)

        os.makedirs(address3, exist_ok=True)
        # 打开可见光图像和边缘图像
        visible_image = Image.open(visible_path).convert("RGB")
        visible_image = auto_tone(visible_image)

        edge_image = Image.open(edge_path).convert("L")  # 转为灰度图像
        
        # 反转边缘图像
        edge_image = ImageOps.invert(edge_image)
        
        # 应用自动对比度
        enhancer = ImageEnhance.Contrast(edge_image)
        edge_image = enhancer.enhance(5)  # 你可以调整2来调整对比度
        
        # 将边缘图像转换为RGB模式，便于操作
        edge_image = edge_image.convert("RGB")
        
        # 进行divide叠加
        result_image = divide_blend(visible_image, edge_image)
        
        # 保存最终的叠加图像
        result_image.save(output_path)
        print(f"Saved: {output_path}")

# 示例调用
address1 = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s2/natural"
address2 = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s2/natural_edge"
address3 = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/train/s2/natural_enhance"
process_images(address1, address2, address3)
