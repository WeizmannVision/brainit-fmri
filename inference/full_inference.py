"""
Full inference pipeline for fMRI-to-image reconstruction.
Combines low-level (VGG) and semantic (diffusion) decoders.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.transform import resize

# Add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2', 'generative_models'))

from utils.datasets import EmbedGraphDataset, collate, DatasetExtWraper
from utils.diffusion_utils import load_diffusion_engine, enhance_recons
from utils.low_level_utils import LowLevelREC
from models.combined_diffusion_engine import CombinedDiffusionEngine
import src.MindEyeV2.generative_models.sgm

# Directory constants
DATA_DIR = 'data/nsd_data/'
DERIVED_DATA_DIR = 'data/derived_data/'
MODEL_DIR = 'results/saved_models/'
OUTPUT_DIR = 'results/reconstructions/'

# Test mode settings
test_samples = 5


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Full inference pipeline')
    parser.add_argument('--run_name', type=str, default='nsd_rec', 
                       help='subdirectory name inside output directory')
    parser.add_argument('--vgg_model', type=str, default='decoder_vgg_ext-1_batch64_save.pth',
                       help='VGG decoder model name')
    parser.add_argument('--clipg_model', type=str, default='decoder_clipg_ext-1_save.pth',
                       help='CLIP-guided decoder model name')
    parser.add_argument('--stage2_model', type=str, default='combined_model.ckpt',
                       help='Stage 2 diffusion model name')
    parser.add_argument('--v2c_mapping', type=str, default=None,
                       help='path to v2c mapping file')
    parser.add_argument('--centers', type=int, default=128,
                       help='number of centers')
    parser.add_argument('--subjects', type=int, nargs='+', default=[0,1,4,6],
                       help='subject indices to process')
    parser.add_argument('--test', action='store_true',
                       help='test mode: only process 5 images of subject 0')
    
    args = parser.parse_args()
    
    # Override settings in test mode
    if args.test:
        args.subjects = [0]
    
    return args


def load_data(args):
    """Load fMRI data and create data structures"""
    print("Loading data...")
    
    # Load subject voxel counts
    num_voxels_subjects = np.load(DATA_DIR + 'num_voxels_all_subjects.npy')
    num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)
    N = num_voxels_subjects.sum()
    
    # Load fMRI data
    file_ = np.load(DATA_DIR + "fmri_v2.npz")
    type_sample = file_["type_sample"]
    multi_sub_fmri = file_['multi_sub_fmri']
    
    # Load v2c mapping
    if args.v2c_mapping is not None:
        v2c_mapping = np.load(args.v2c_mapping)
    else:
        v2c_mapping = np.load(DERIVED_DATA_DIR + "v2c_128_mapping_gmm_v2.npy")
    
    # Load ground truth images (if available)
    imgs_path = DATA_DIR + "all_images_v2.npy"
    imgs = None
    if os.path.exists(imgs_path):
        imgs = np.load(imgs_path)
        imgs = imgs[type_sample == 2]
    
    return {
        'num_voxels_subjects': num_voxels_subjects,
        'N': N,
        'multi_sub_fmri': multi_sub_fmri,
        'v2c_mapping': v2c_mapping,
        'imgs': imgs
    }


def load_models(args, device):
    """Load all models"""
    print("Loading models...")
    
    # VGG decoder (low-level)
    print(f"  Loading VGG decoder: {args.vgg_model}")
    vgg_model = torch.load(os.path.join(MODEL_DIR, args.vgg_model))
    vgg_model = vgg_model.eval().to(device).float()
    
    # Load diffusion engine
    print(f"  Loading diffusion engine...")
    diffusion_engine = load_diffusion_engine()
    
    # Load CLIP-guided decoder
    print(f"  Loading CLIP-guided decoder: {args.clipg_model}")
    gnn_model = torch.load(os.path.join(MODEL_DIR, args.clipg_model))
    
    # Load combined stage 2 model
    print(f"  Loading stage 2 model: {args.stage2_model}")
    stage2_path = os.path.join(MODEL_DIR, args.stage2_model)
    combined_model = CombinedDiffusionEngine.load_from_checkpoint(
        stage2_path, 
        map_location="cpu",
        diffusion_engine=diffusion_engine, 
        gnn_model=gnn_model
    ).to(device)
    
    return {
        'vgg_model': vgg_model,
        'combined_model': combined_model
    }


def get_vgg_predictions(vgg_model, data, args, device, num_samples):
    """Get predictions from VGG decoder"""
    print("Getting VGG predictions...")
    
    num_voxels_subjects = data['num_voxels_subjects']
    multi_sub_fmri = data['multi_sub_fmri']
    v2c_mapping = data['v2c_mapping']
    
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects
    
    embeds_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        
        sub_vector = np.ones(num_samples, dtype=int) * sub
        Y_placeholder = np.zeros((num_samples, 1))
        
        test_loader = EmbedGraphDataset(
            multi_sub_fmri[:num_samples, start[sub]:end[sub]],
            Y_placeholder, 
            v2c_mapping, 
            sub_vector,
            sub_num_voxels=num_voxels_subjects, 
            sample=False, 
            num_centers=args.centers
        )
        
        predicts = []
        for i in range(num_samples):
            fmri, _, sub_idx, indexes = test_loader[i]
            
            with torch.no_grad():
                predict = vgg_model(
                    fmri.unsqueeze(0).to(device), 
                    sub_idx.to(device),
                    indexes.to(device)
                ).detach().cpu()
            predicts.append(predict)
        
        predicts_test = np.stack(predicts, axis=0)
        embeds_sub[f"sub_{sub}"] = predicts_test
        print(f"    Shape: {predicts_test.shape}")
    
    return embeds_sub


def reconstruct_lowlevel_images(embeds_sub, args, num_samples):
    """Reconstruct low-level images from VGG embeddings"""
    print("Reconstructing low-level images...")
    
    lw_rec = LowLevelREC()
    imgs_lw_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        imgs = []
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"    Sample {i}/{num_samples}")
            img_i = lw_rec.rec_image(embeds_sub[f"sub_{sub}"][i])
            imgs.append(img_i)
        imgs_lw_sub[sub] = np.stack(imgs)
        print(f"    Output shape: {imgs_lw_sub[sub].shape}")
    
    return imgs_lw_sub


def reconstruct_semantic_images(combined_model, imgs_lw_sub, data, args, device, num_samples):
    """Reconstruct semantic images using diffusion model"""
    print("Reconstructing semantic images with diffusion model...")
    
    num_voxels_subjects = data['num_voxels_subjects']
    multi_sub_fmri = data['multi_sub_fmri']
    v2c_mapping = data['v2c_mapping']
    
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects
    
    imgs_semantic_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        
        sub_vector = np.ones(num_samples, dtype=int) * sub
        Y_placeholder = np.zeros((num_samples, 1))
        
        test_loader = EmbedGraphDataset(
            multi_sub_fmri[:num_samples, start[sub]:end[sub]],
            Y_placeholder,
            v2c_mapping,
            sub_vector,
            sub_num_voxels=num_voxels_subjects,
            sample=False,
            num_centers=args.centers
        )
        
        imgs = []
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"    Sample {i}/{num_samples}")
            
            fmri, _, sub_idx, indexes = test_loader[i]
            
            with torch.no_grad():
                img = combined_model.generate(
                    fmri.unsqueeze(0).cuda(),
                    sub_idx.cuda(),
                    indexes.cuda(),
                    img_init=imgs_lw_sub[sub][i]
                ).detach().cpu()
            
            imgs.append(img)
        imgs = np.concatenate(imgs,axis=0)
        imgs_semantic_sub[sub] = np.transpose(imgs,[0,2,3,1])  
        print(f"    Output shape: {imgs_semantic_sub[sub].shape}")
    
    return imgs_semantic_sub


def enhance_semantic_images(imgs_semantic_sub, args):
    """Enhance semantic images using SDXL"""
    print("Enhancing semantic images with SDXL...")
    
    imgs_enhanced_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        imgs_enhanced_sub[sub] = enhance_recons(imgs_semantic_sub[sub])
        print(f"    Shape: {imgs_enhanced_sub[sub].shape}")
    
    return imgs_enhanced_sub


def save_results(imgs_lw, imgs_semantic, imgs_enhanced, data, args, num_samples):
    """Save reconstruction results"""
    print("Saving results...")
    
    output_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for sub in args.subjects:
        sub_dir = os.path.join(output_dir, f"subject_{sub}")
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save low-level reconstructions
        if imgs_lw is not None and sub in imgs_lw:
            imgs_uint8 = (imgs_lw[sub] * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "low_level_recons.npy"), imgs_uint8)
            lw_dir = os.path.join(sub_dir, "low_level")
            os.makedirs(lw_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(lw_dir, f"img_{i:04d}.png"), imgs_uint8[i])
        
        # Save semantic reconstructions
        if imgs_semantic is not None and sub in imgs_semantic:
            imgs_uint8 = (imgs_semantic[sub] * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "semantic_recons.npy"), imgs_uint8)
            semantic_dir = os.path.join(sub_dir, "semantic")
            os.makedirs(semantic_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(semantic_dir, f"img_{i:04d}.png"), imgs_uint8[i])
        
        # Save enhanced reconstructions
        if imgs_enhanced is not None and sub in imgs_enhanced:
            imgs_resized = np.array([resize(imgs_enhanced[sub][i], (224, 224), anti_aliasing=True) for i in range(num_samples)])
            imgs_uint8 = (imgs_resized * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "enhanced_recons.npy"), imgs_uint8)
            enhanced_dir = os.path.join(sub_dir, "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(enhanced_dir, f"img_{i:04d}.png"), imgs_uint8[i])
    
    print(f"Results saved to {output_dir}")


def main():
    """Main inference pipeline"""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(args)
    
    # Determine number of samples
    if args.test:
        num_samples = test_samples
    else:
        num_samples = data['multi_sub_fmri'].shape[0]
    
    # Load models
    models = load_models(args, device)
    
    # Get VGG predictions
    embeds_sub = get_vgg_predictions(models['vgg_model'], data, args, device, num_samples)
    
    # Reconstruct low-level images
    imgs_lw_sub = reconstruct_lowlevel_images(embeds_sub, args, num_samples)
    
    # Reconstruct semantic images with diffusion model
    imgs_semantic_sub = reconstruct_semantic_images(models['combined_model'], imgs_lw_sub, data, args, device, num_samples)
    
    # Enhance semantic images
    imgs_enhanced_sub = enhance_semantic_images(imgs_semantic_sub, args)
    
    # Save results
    save_results(imgs_lw_sub, imgs_semantic_sub, imgs_enhanced_sub, data, args, num_samples)
    
    print("Inference complete!")


if __name__ == "__main__":
    main()
