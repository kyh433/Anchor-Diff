import os
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy.linalg import sqrtm
from scipy.stats import entropy
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import normalized_mutual_info_score

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Helper function to load images from a folder
def load_images_from_folder(folder_path, file_type=None):
    images = []
    filenames = []
    image_files = sorted(os.listdir(folder_path))
    
    for image_file in image_files:
        if image_file.endswith(".png"):
            if file_type and file_type in image_file:  # Filter by "edit" or "reconstruct"
                img_path = os.path.join(folder_path, image_file)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                filenames.append(image_file)  # Save the filename
            elif not file_type:  # If no file_type is provided, load all images
                img_path = os.path.join(folder_path, image_file)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                filenames.append(image_file)  # Save the filename
    return images, filenames

# Helper function to load text from a jsonl file
def load_text_from_jsonl(text_file):
    with open(text_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    # Extract just the 'text' field from each entry
    texts = [entry['text'] for entry in data]
    return texts

########### FID ###############
# InceptionV3 Model for FID calculation
def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    
    model.fc = torch.nn.Identity()
    
    model.eval()
    return model

# Preprocess image for InceptionV3, ensuring it has 3 channels (RGB)
def preprocess_image(image):
    # Convert grayscale to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize image to 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Feature extraction
def extract_features(images, model):
    features = []
    for img in images:
        img_tensor = preprocess_image(img).cuda()  # Convert image to tensor and move to GPU
        
        # Print the image tensor shape to debug
        # print(f"Image tensor shape before model: {img_tensor.shape}")
        
        with torch.no_grad():
            # print(model)
            feature = model(img_tensor)  # Get feature from intermediate layer
            
            # Print the feature tensor shape to debug
            # print(f"Feature tensor shape before pooling: {feature.shape}")
            
            # InceptionV3 outputs features with shape [1, 2048, 1, 1], we don't need additional pooling
            feature = feature.squeeze().cpu().numpy()  # Remove extra dimensions
            features.append(feature)
    
    return np.array(features)

# FID Calculation
def calculate_fid(real_images, generated_images, model):
    real_features = extract_features(real_images, model)
    generated_features = extract_features(generated_images, model)

    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_gen
    cov_sqrt = sqrtm(cov_real @ cov_gen)
    fid = diff @ diff.T + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return fid

############## IS #################
def get_inception_model_for_is():
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model

# Calculate Inception Score
def calculate_inception_score(images, model, splits=10):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get class probabilities for each image
    preds = []
    for img in images:
        img_tensor = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img_tensor)
            preds.append(F.softmax(output, dim=1).cpu().numpy())

    preds = np.array(preds)
    
    # Calculate IS for each split
    split_scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits: (i + 1) * len(preds) // splits]
        kl_div = entropy(np.mean(part, axis=0), part.T)
        is_score = np.exp(np.mean(kl_div))
        split_scores.append(is_score)

    return np.mean(split_scores), np.std(split_scores)

############### CLIP ######################
def get_clip_model():
    model = CLIPModel.from_pretrained("/mnt/data/project_kyh/weight/openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("/mnt/data/project_kyh/weight/openai/clip-vit-large-patch14")
    return model, processor

def calculate_clip_similarity(images, texts, model, processor):
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to('cuda')
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to('cuda')

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

    # Normalize features
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).cpu().numpy()
    return similarity

############## NMI ####################
def load_and_convert_image(image_path):
    image = Image.open(image_path).convert("L")  # "L" mode for grayscale
    return np.array(image)  # Convert to NumPy array

def calculate_nmi(image1, image2):
    return normalized_mutual_info_score(image1.ravel(), image2.ravel())

# Function to calculate metrics for all images in the folder
def calculate_metrics_for_folder(real_nat_folder,real_vh_folder, generated_folder, text_file, clip_model, processor):
    # Load images
    real_nat_images,real_nat_filenames = load_images_from_folder(real_nat_folder)
    real_vh_images,real_vh_filenames = load_images_from_folder(real_vh_folder)
    generated_images_edit,generated_images_edit_filenames = load_images_from_folder(generated_folder, file_type="edit")
    generated_images_reconstruct,generated_images_reconstruct_filenames = load_images_from_folder(generated_folder, file_type="reconstruct")
    
    # # Check the filenames for consistency
    # print("Real Nat Image Filenames:", real_nat_filenames[:5])  # Display the first 5 filenames
    # print("Real VH Image Filenames:", real_vh_filenames[:5])    # Display the first 5 filenames
    # print("Generated Edit Image Filenames:", generated_images_edit_filenames[:5])  # Display the first 5 filenames
    # print("Generated Reconstruct Image Filenames:", generated_images_reconstruct_filenames[:5])  # Display the first 5 filenames
    # import matplotlib.pyplot as plt

    # # Function to display images in a grid
    # def display_images(images, title="Images"):
    #     plt.figure(figsize=(10, 10))
    #     num_images = len(images)
    #     cols = 5  # Number of columns in the grid
    #     rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed
        
    #     for i, img in enumerate(images):
    #         plt.subplot(rows, cols, i + 1)
    #         plt.imshow(img)
    #         plt.axis("off")
    #         plt.title(f"Image {i+1}")
    #     plt.suptitle(title)
    #     plt.show()

    # # Display some of the real and generated images for verification
    # real_nat_images_sample = real_nat_images[:5]  # Select the first 5 real images
    # real_vh_images_sample = real_vh_images[:5]  # Select the first 5 real images
    # generated_images_edit_sample = generated_images_edit[:5]  # Select the first 5 generated SAR images
    # generated_images_reconstruct_sample = generated_images_reconstruct[:5]  # Select the first 5 generated Visible images

    # # Display the images
    # display_images(real_nat_images_sample, title="Real Optica Images (Optical)")
    # display_images(real_vh_images_sample, title="Real SAR Images (SAR)")
    # display_images(generated_images_reconstruct_sample, title="Generated Optical Images (Optical)")
    # display_images(generated_images_edit_sample, title="Generated SAR Images (SAR)")



    texts = load_text_from_jsonl(text_file)

    # Get model for FID and IS
    fid_model = get_inception_model().cuda()
    is_model = get_inception_model_for_is().cuda()

    # FID score



    fid_score_nat = calculate_fid(real_nat_images, generated_images_reconstruct, fid_model)
    fid_score_vh = calculate_fid(real_vh_images, generated_images_edit, fid_model)
    print(f"FID Score (Optical): {fid_score_nat}")
    print(f"FID Score (SAR): {fid_score_vh}")

    # Inception Score
    is_score_nat, is_std_nat = calculate_inception_score(generated_images_reconstruct, is_model)
    is_score_vh, is_std_vh = calculate_inception_score(generated_images_edit, is_model)
    print(f"Inception Score (Optical): {is_score_nat} ± {is_std_nat}")
    print(f"Inception Score (SAR): {is_score_vh} ± {is_std_vh}")

    # CLIP similarity
    similarity = calculate_clip_similarity(generated_images_reconstruct, texts, clip_model, processor)
    print(f"CLIP similarity: {similarity.mean()}")

    # NMI between real and generated images
    for edit_image, reconstruct_image in zip(generated_images_edit, generated_images_reconstruct):
        nmi_score = calculate_nmi(np.array(edit_image.convert("L")), np.array(reconstruct_image.convert("L")))
        print(f"NMI Score: {nmi_score}")

# Example usage:
real_nat_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s2/natural"
real_vh_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s1/vh"
generated_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/beta_output/natPCA2s1_vh_attention"
text_file = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/split_by_modality_Hunan/test/s2_nat.jsonl"

# Get CLIP model
clip_model, processor = get_clip_model()

# Calculate all metrics for the folder
calculate_metrics_for_folder(real_nat_folder,real_vh_folder, generated_folder, text_file, clip_model, processor)
