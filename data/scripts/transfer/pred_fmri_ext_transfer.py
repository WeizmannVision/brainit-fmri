#!/usr/bin/env python3
"""
Script to predict synthetic fMRI from external images using trained encoder.
Transfer learning version for data/transfer pipeline.
"""

import sys
import os
import torch 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
device = torch.device("cuda")

TRANSFER_SUB = "transfer_sub"


def trans_imgs_shift(img):
    """
    Transform and normalize images for encoder model.
    
    Args:
        img: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        torch tensor of shape (C, H, W) normalized for model input
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)
    img = (img - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img.astype(float)).float()
    return img


def main():
    # Load trained transfer learning encoder model
    print("Loading transfer learning encoder model...")
    encoder_model = torch.load(f"results/saved_models/encoder_{TRANSFER_SUB}.pth").eval()
    encoder_model = encoder_model.to(device)
    print("  Model loaded!")
    
    # Paths
    data_dir = "data/transfer/"
    
    # Load external images (224x224)
    ext_imgs_path = data_dir + "ext_images_224.npy"
    
    if not os.path.exists(ext_imgs_path):
        print(f"Error: {ext_imgs_path} not found!")
        print("Please ensure external images are available for synthetic fMRI generation.")
        return
    
    print(f"\nLoading external images from {ext_imgs_path}...")
    ext_imgs = np.load(ext_imgs_path)
    print(f"  Shape: {ext_imgs.shape}, dtype: {ext_imgs.dtype}")
    
    # Get number of voxels for this subject (see TRANSFER_SUB)
    fmri_data = np.load(data_dir + f"{TRANSFER_SUB}_fmri.npz")
    num_voxels = fmri_data['train'].shape[1]
    print(f"\nNumber of voxels: {num_voxels}")
    
    # Predict fMRI for external images
    print(f"\nPredicting fMRI for {ext_imgs.shape[0]} images...")
    embeds = np.zeros([ext_imgs.shape[0], num_voxels], dtype=np.float32)
    
    for i in range(ext_imgs.shape[0]):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{ext_imgs.shape[0]}")
        
        # Preprocess image
        image_tensor = trans_imgs_shift(ext_imgs[i] / 255.0).unsqueeze(0)
        
        # Predict fMRI
        with torch.no_grad():
            pred = encoder_model(image_tensor.cuda(), torch.arange(num_voxels).unsqueeze(0))
        
        embeds[i] = pred.detach().cpu().numpy()
    
    # Save predicted fMRI
    output_dir = "data/derived_data/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + f"ext_fmri_{TRANSFER_SUB}.npy"
    
    print(f"\nSaving to {output_path}...")
    np.save(output_path, embeds.astype(np.float16))
    print(f"  Saved shape: {embeds.shape}, dtype: float16")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
