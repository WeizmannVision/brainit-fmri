#!/usr/bin/env python3
"""
Script to extract CLIP embeddings from image arrays.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import open_clip
import torch
import kornia


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
    # Paths - nsd_data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nsd_data')
    
    input_files = [
        'all_images_v2_224.npy',
        'ext_images_224.npy'
    ]
    
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
    
    for input_file in input_files:
        input_path = os.path.join(data_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        # Load images
        print(f"Loading {input_file}...")
        images = np.load(input_path)
        print(f"  Shape: {images.shape}, dtype: {images.dtype}")
        
        # Extract CLIP embeddings
        clip_embeddings = extract_clip_embeddings(images, model_clip)
        
        # Save embeddings
        base_name = input_file.replace('_224.npy', '')
        output_file = f"{base_name}_clip.npy"
        output_path = os.path.join(data_dir, output_file)
        
        print(f"Saving to {output_file}...")
        np.save(output_path, clip_embeddings)
        print(f"  Saved shape: {clip_embeddings.shape}, dtype: {clip_embeddings.dtype}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
