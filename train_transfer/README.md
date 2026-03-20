# Transfer Learning

This directory contains scripts for adapting pretrained models to new subjects/datasets.

## Overview

All transfer learning steps adapt the voxel embedding of trained models. The key aspects are:

- **Voxel-to-cluster mapping**: Mapping of new voxels to existing cluster centers is established according to nearest neighbor (NN) in the embedding space
- **Voxel embedding initialization**: The initialization of voxel embeddings in transfer learning uses NN voxel embeddings from the reference model (this works better than random initialization)

## Data Preparation

Before training, prepare the transfer learning data:

```bash
# Prepare images at different resolutions
python data/scripts/transfer/prepare_imgs_transfer.py

# Extract CLIP embeddings
python data/scripts/transfer/prepare_clip_transfer.py
```

## Training

### 1. Train Encoder
```bash
python train_transfer/train_encoder_transfer.py
```

### 2. Map Voxel Clusters and Generate Synthetic fMRI
After training the encoder, map voxels to clusters using NN and establish NN for voxel embedding init:
```bash
# Map voxels to clusters
python data/scripts/transfer/map_clusters_transfer.py

# Generate synthetic fMRI
python data/scripts/transfer/pred_fmri_ext_transfer.py
```

### 3. Train Decoder
```bash
# VGG-based decoder with contrastive loss
python train_transfer/train_decoder_transfer.py --VGG --CONT --EXT --SAVE

# CLIP-guided decoder
python train_transfer/train_decoder_transfer.py --CLIPG --EXT --SAVE
```

### 4. Train Decoder Stage 2
```bash
python train_transfer/train_decoder_stage2_transfer.py --EXT
```
