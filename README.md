# Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer

Implementation of:

- **The Wisdom of a Crowd of Brains: A Universal Brain Encoder** — Roman Beliy\*, Navve Wasserman\*, Amit Zalcher, Michal Irani. [arXiv:2406.12179](https://arxiv.org/abs/2406.12179)

- **Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer** — Roman Beliy\*, Amit Zalcher\*, Jonathan Kogman, Navve Wasserman, Michal Irani. Accepted at **ICLR 2026**. [arXiv:2510.25976](https://arxiv.org/abs/2510.25976)

\* Stands for equal contribution.

## Requirements

Environment requirements are in `env.yml`. To create the conda environment:
```bash
conda env create -f env.yml
conda activate brain-it
```


## Overview
This repository implements the **Universal Brain Encoder** (image-to-fMRI encoding) and **Brain-IT** (fMRI-to-image reconstruction with the Brain-Interaction Transformer), as described in the papers above.

## Roadmap

- [ ] **NSD data preparation** — end-to-end scripts and documentation for preparing inputs from [Natural Scenes Dataset](https://naturalscenesdataset.org/) (NSD) data
- [ ] **Model checkpoints** — published pretrained weights and instructions for where to place them under `results/saved_models/`
- [ ] **External models** — links and setup for third-party models
- [ ] **Transfer learning** — full training code and configs for adapting Brain-IT to new subjects or datasets.

## Quick Start

For inference on pretrained models, please run:
```bash
python inference/full_inference.py
```


## Directory Structure
```
Brain-IT/
├── data/
│   ├── nsd_data/              # NSD dataset files
│   ├── derived_data/          # Derived data (clusters, embeddings)
│   ├── external_models/       # External pretrained models
│   └── scripts/               # Data processing scripts
├── models/                    # Model architectures
├── train/                     # Training scripts
├── inference/                 # Inference scripts
├── utils/                     # Utility functions
└── results/                   # Output directory
    ├── saved_models/          # Trained model checkpoints
    └── reconstructions/       # Inference outputs
```

## Data Preparation

### 1. Prepare Images
```bash
python data/scripts/prepare_imgs.py
```

### 2. Extract CLIP Embeddings
```bash
python data/scripts/prepare_clip.py
```

## Training

### 1. Train Encoder
```bash
python train/train_encoder.py
```

### 2. Generate Voxel Clusters and Synthetic fMRI
After training the encoder, map voxels to clusters and generate synthetic fMRI:
```bash
# Map voxels to clusters
python data/scripts/get_clusters.py

# Generate synthetic fMRI
python data/scripts/pred_fmri_ext.py
```

### 3. Train Decoder
```bash
# VGG-based decoder with contrastive loss
python train/train_decoder.py --VGG --CONT --EXT --SAVE

# CLIP-guided decoder (stage 1)
python train/train_decoder.py --CLIPG --EXT --SAVE
```

### 4. Train Decoder Stage 2 (Diffusion)
```bash
python train/train_decoder_stage2.py --EXT
```

**Note:** Stage 2 training requires significant GPU memory:
- **2x H200 GPUs**, or **4x H100 GPUs** (if 4 H100 are used set batch size to 4: modify `BATCH_SIZE = 4` in the script)

## Inference 

Run the full inference pipeline:
```bash
python inference/full_inference.py --run_name my_experiment
```


### Output Structure:
Results are saved in `results/reconstructions/{run_name}/`:
```
results/reconstructions/{run_name}/
└── subject_{n}/
    ├── low_level/              # Low-level VGG reconstructions
    │   └── img_*.png
    ├── semantic/               # Semantic diffusion reconstructions
    │   └── img_*.png
    ├── enhanced/               # Enhanced SDXL reconstructions (224x224)
    │   └── img_*.png
    ├── low_level_recons.npy    # Full arrays (uint8)
    ├── semantic_recons.npy
    └── enhanced_recons.npy
```

## License

This code accompanies the arXiv preprints linked at the top of this README. The PDFs are shared under the [license stated on each arXiv record](https://arxiv.org/help/license) (see the “license” icon on the abstract pages). If you use this code, please cite those papers. Third-party or vendored code may have its own terms—check the relevant subdirectories (e.g. model bundles under `src/`).

## Contact

For questions or inquiries: [roman.beliy@weizmann.ac.il](mailto:roman.beliy@weizmann.ac.il).

