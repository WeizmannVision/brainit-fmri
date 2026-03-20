#!/usr/bin/env python3
"""
Script to prepare image arrays at different resolutions (112 and 224) from 256x256 images.
"""

import numpy as np
from skimage.transform import resize
import os


def resize_images(images, target_size):
    """
    Resize a batch of images to target size.
    
    Args:
        images: numpy array of shape (N, H, W, 3) in uint8
        target_size: tuple (height, width) for target resolution
        
    Returns:
        numpy array of resized images in uint8
    """
    n_images = images.shape[0]
    resized = np.zeros((n_images, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    print(f"Resizing {n_images} images to {target_size}...")
    
    for i in range(n_images):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{n_images}")
        
        # Resize with preserve_range=True to keep 0-255 range (not 0-1)
        # Output will be float in range [0, 255]
        # anti_aliasing=True to prevent aliasing artifacts
        resized_img = resize(images[i], target_size, 
                            preserve_range=True, 
                            anti_aliasing=True)
        
        # Round, clip to valid uint8 range, and convert to uint8
        resized[i] = np.clip(np.round(resized_img), 0, 255).astype(np.uint8)
    
    return resized


def main():
    # Paths - nsd_data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nsd_data')
    
    input_files = [
        'all_images_v2_256.npy',
        'ext_images_256.npy'
    ]
    
    target_resolutions = [112, 224]
    
    for input_file in input_files:
        input_path = os.path.join(data_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        # Load original images
        print(f"Loading {input_file}...")
        images_256 = np.load(input_path)
        print(f"  Shape: {images_256.shape}, dtype: {images_256.dtype}")
        
        # Generate base filename without extension
        base_name = input_file.replace('_256.npy', '')
        
        # Resize to each target resolution
        for target_res in target_resolutions:
            print(f"\nResizing to {target_res}x{target_res}...")
            resized_images = resize_images(images_256, (target_res, target_res))
            
            # Save resized images
            output_file = f"{base_name}_{target_res}.npy"
            output_path = os.path.join(data_dir, output_file)
            
            print(f"Saving to {output_file}...")
            np.save(output_path, resized_images)
            print(f"  Saved shape: {resized_images.shape}, dtype: {resized_images.dtype}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
