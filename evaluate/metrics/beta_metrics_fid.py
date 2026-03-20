import os
import torch
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models
from torch import nn

# Step 1: Load images from folder
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

# Step 2: Preprocess input images for Inception v3
def prepare_inception_input(images, device):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images to Inception v3 input size
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])  # Define preprocessing steps for Inception v3 input

    # Preprocess each image and move to device (GPU/CPU)
    preprocessed_images = [preprocess(img).unsqueeze(0).to(device) for img in images]
    return torch.cat(preprocessed_images, dim=0)  # Combine into a single tensor

# Step 3: Load Inception v3 model and remove the fully connected layer
def load_inception_model(device):
    model = models.inception_v3(pretrained=True, transform_input=False)  # Load pre-trained Inception v3 model without input transformation
    model.fc = nn.Identity()  # Remove the fully connected layer (classifier)
    model.eval()  # Set the model to evaluation mode
    return model.to(device)  # Move the model to the same device as our GAN

# Step 4: Extract features using Inception v3
def extract_features(images, model, device):
    with torch.no_grad():  # Disable gradient computation for efficiency
        features = model(images)  # Extract features
    return features.cpu().numpy()  # Convert features to numpy for further processing

# Step 5: Calculate the mean and covariance of features
def calculate_mean_and_covariance(features):
    mean = np.mean(features, axis=0)  # Calculate mean across all samples for each feature
    covariance = np.cov(features, rowvar=False)  # Calculate covariance matrix of features
    return mean, covariance

# Step 6: Calculate FID score
def calculate_frechet_inception_distance(real_mean, real_cov, generated_mean, generated_cov):
    """Calculate the Fréchet Inception Distance (FID) between real and generated image features."""
    # Calculate squared L2 norm between means
    mean_diff = np.sum((real_mean - generated_mean) ** 2)

    # Calculate sqrt of product of covariances
    covmean = sqrtm(real_cov.dot(generated_cov))

    # Check and correct imaginary parts if necessary
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Take only the real part if result is complex

    # Calculate trace term
    trace_term = np.trace(real_cov + generated_cov - 2 * covmean)

    # Compute FID
    fid = mean_diff + trace_term

    return fid  # Return FID as a Python float

# Step 7: Main function to compute FID
def compute_fid(real_nat_folder, real_vh_folder, generated_folder, device):
    # Load images
    real_nat_images, _ = load_images_from_folder(real_nat_folder)
    real_vh_images, _ = load_images_from_folder(real_vh_folder)

    val_nat_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s2/natural"
    val_vh_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/val/s1/vh"
    
    generated_images_reconstruct, _ = load_images_from_folder(val_nat_folder)
    generated_images_edit, _ = load_images_from_folder(val_vh_folder)
    # generated_images_edit, _ = load_images_from_folder(generated_folder, file_type="edit")
    # generated_images_reconstruct, _ = load_images_from_folder(generated_folder, file_type="reconstruct")

    # Load the Inception v3 model
    inception_model = load_inception_model(device)

    # Preprocess images and extract features
    real_nat_input = prepare_inception_input(real_nat_images, device)
    real_vh_input = prepare_inception_input(real_vh_images, device)
    generated_input_edit = prepare_inception_input(generated_images_edit, device)
    generated_input_reconstruct = prepare_inception_input(generated_images_reconstruct, device)

    # Extract features
    real_nat_features = extract_features(real_nat_input, inception_model, device)
    real_vh_features = extract_features(real_vh_input, inception_model, device)
    generated_edit_features = extract_features(generated_input_edit, inception_model, device)
    generated_reconstruct_features = extract_features(generated_input_reconstruct, inception_model, device)

    # # Combine features (real and generated)
    # all_real_features = np.concatenate([real_nat_features, real_vh_features], axis=0)
    # all_generated_features = np.concatenate([generated_edit_features, generated_reconstruct_features], axis=0)

    # Calculate mean and covariance for real and generated features
    real_nat_mean, real_nat_cov = calculate_mean_and_covariance(real_nat_features)
    generated_nat_mean, generated_nat_cov = calculate_mean_and_covariance(generated_reconstruct_features)

    # Calculate mean and covariance for real and generated features
    real_vh_mean, real_vh_cov = calculate_mean_and_covariance(real_vh_features)
    generated_vh_mean, generated_vh_cov = calculate_mean_and_covariance(generated_edit_features)

    # Calculate FID
    fid_nat_score = calculate_frechet_inception_distance(real_nat_mean, real_nat_cov, generated_nat_mean, generated_nat_cov)
    print(f"natFID Score: {fid_nat_score:.4f}")

    fid_vh_score = calculate_frechet_inception_distance(real_vh_mean, real_vh_cov, generated_vh_mean, generated_vh_cov)
    print(f"vhFID Score: {fid_vh_score:.4f}")
    return 0

# Step 8: Example usage
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
real_nat_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s2/natural"
real_vh_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/traindata/Processed_Hunan_Dataset/test/s1/vh"
generated_folder = "/mnt/data/project_kyh/MultimodalityGeneration25/beta_output/natPCA2s1_vh_attention"

compute_fid(real_nat_folder, real_vh_folder, generated_folder, device)
