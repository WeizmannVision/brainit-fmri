#!/usr/bin/env python3
"""
Script to extract CLIP embeddings from image arrays.
Transfer learning version for data/transfer pipeline.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import open_clip
import torch
import kornia

TRANSFER_SUB = "transfer_sub"


def preprocess(x):
    """
    Preprocess image for CLIP model.
    
    Args:
        x: tensor in range [0, 1] with shape (C, H, W)
    
    Returns:
        preprocessed tensor
    """
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
    x = (x + 1.0) / 2.0  # not a bug: required for this open_clip / CLIP preprocessing before ImageNet normalize
    # renormalize according to clip
    x = kornia.enhance.normalize(x, mean, std)
    return x


def extract_clip_embeddings(images, model_clip):
    """
    Extract CLIP embeddings for images.
    
    Args:
        images: numpy array of shape (N, H, W, 3) in uint8
        model_clip: CLIP model
        
    Returns:
        numpy array of shape (N, 256, 1664) containing CLIP embeddings
    """
    n_images = images.shape[0]
    
    # Preallocate memory for embeddings
    all_tokens = np.zeros((n_images, 256, 1664), dtype=np.float16)
    
    print(f"Extracting CLIP embeddings for {n_images} images...")
    
    with torch.no_grad():
        for i in range(n_images):
            if i % 1000 == 0:
                print(f"  Processing image {i}/{n_images}")
            
            # Convert to tensor and normalize to [0, 1]
            img_tensor = torch.tensor(images[i] / 255.0).permute(2, 0, 1).unsqueeze(0)
            img_preprocessed = preprocess(img_tensor).to('cuda')
            
            # Extract features
            x, tokens = model_clip.visual(img_preprocessed.float())
            
            # Store tokens
            all_tokens[i] = tokens.cpu().numpy().astype(np.float16)
    
    print(f"  Final shape: {all_tokens.shape}")
    
    return all_tokens


def main():
    data_dir = "data/transfer"
    
    # Input file (224x224 images)
    input_file = f"{TRANSFER_SUB}_imgs_224.npz"
    input_path = os.path.join(data_dir, input_file)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        print("Please run prepare_imgs_transfer.py first to generate 224x224 images.")
        return
    
    # Initialize CLIP model
    print("Loading CLIP model...")
    output_tokens = True
    model_clip, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-bigG-14",
        device=torch.device('cuda'),
        pretrained="laion2b_s39b_b160k",
    )
    model_clip.visual.output_tokens = output_tokens
    model_clip.eval()
    print("  Model loaded!")
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    # Load images (train only)
    print(f"Loading {input_file}...")
    data = np.load(input_path)
    train_images = data['train']
    print(f"  Train shape: {train_images.shape}, dtype: {train_images.dtype}")
    
    # Extract CLIP embeddings for train set
    print(f"\n{'='*60}")
    print("Processing train set...")
    print(f"{'='*60}")
    train_clip = extract_clip_embeddings(train_images, model_clip)
    
    # Save embeddings
    output_file = f"{TRANSFER_SUB}_imgs_clip.npy"
    output_path = os.path.join(data_dir, output_file)
    
    print(f"\nSaving to {output_file}...")
    np.save(output_path, train_clip)
    print(f"  Saved train shape: {train_clip.shape}, dtype: {train_clip.dtype}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
