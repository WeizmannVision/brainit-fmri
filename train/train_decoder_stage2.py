import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import random
from sklearn.model_selection import train_test_split
from functools import partial
import wandb
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.loggers import TensorBoardLogger

#from finetune_utils import *

# Set GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #if 4 gpus are used set to 0,1,2,3


sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2','generative_models'))

from utils.datasets import EmbedGraphDataset, collate, DatasetExtWraper
from models.combined_diffusion_engine import CombinedDiffusionEngine
from utils.diffusion_utils import load_diffusion_engine

import src.MindEyeV2.generative_models.sgm
from src.MindEyeV2.generative_models.sgm.models.diffusion_no_lightning import DiffusionEngine_nolightning
from src.MindEyeV2.generative_models.sgm.util import instantiate_from_config
from models.decoder_models import dec_param, Decoder
from torch_geometric.nn.conv import TransformerConv

from pytorch_lightning.callbacks import ModelCheckpoint

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--SAVE', dest='save', action='store_const', const=True, default=False, help='save model')
    parser.add_argument('--EXT', dest='ext', action='store_const', const=True, default=False, help='include external data')
    parser.add_argument('--SAMPLE', dest='ext_sample_factor', type=int, default=-1, help='external freq sample')
    parser.add_argument('--GNN_MODEL_PATH', dest='gnn_model_path', type=str, default=None, help='path to trained GNN model')
    parser.add_argument('--MAP_FILE', dest='v2c_mapping', type=str, default=None, help='path to map file')
    parser.add_argument('--NUM_VOX', dest='num_vox', type=int, default=15, help='number of voxels')
    parser.add_argument('--C', dest='centers', type=int, default=128, help='number of centers')
    parser.add_argument('--NUM_EPOCHS', dest='num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--NUM_GPUS', dest='num_gpus', type=int, default=2, help='number of GPUs to use')

    #parser.add_argument('--SAVE_EVERY_N_EPOCHS', dest='save_every_n_epochs', type=int, default=1, help='save every n epochs')
    return parser.parse_args()

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

diffusion_model_path = "data/external_models/MindEyeV2/unclip6_epoch0_step110000.ckpt"
diffusion_model_cfg = "data/external_models/MindEyeV2/unclip6.yaml"
save_dir = "results/saved_models/"
data_dir = "data/nsd_data/"
derived_data_dir = "data/derived_data/"
tensorbaord_dir =  'logs/tensorboard/decoder_stage2/'

args = parse_arguments()
# Control variable for all saves/outputs

# Training hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 4 if args.num_gpus == 4 else 8
#SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
ACCUMULATE_GRAD_BATCHES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize([0.5], [0.5])
])



name = "decoder_stage2"
      
if(args.ext):
    name+="_ext"+str(args.ext_sample_factor)




# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

def print_gpu_memory_usage():
    """Print GPU memory usage for all available GPUs"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total", flush=True)



# =============================================================================
# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function"""
    
    #################################################     load data ###################################################
    
    num_voxels_subjects = np.load(data_dir + 'num_voxels_all_subjects.npy')
    num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)
    N = num_voxels_subjects.sum()

    file_ = np.load(data_dir + "fmri_v2.npz")
    type_sample = file_["type_sample"]
    single_sub = file_['single_sub']
    single_sub_fmri = file_['single_sub_fmri']
    multi_sub_fmri = file_['multi_sub_fmri']


    val_ind = file_['val_single_ind']
    train_ind = np.ones(single_sub_fmri.shape[0], dtype=bool)
    train_ind[val_ind] = False
    if(args.ext):  
        fmri_ext = np.load(derived_data_dir + "ext_fmri.npy")


    images = np.load(data_dir + "all_images_v2_256.npy")
    images_single = images[type_sample == 1]

    if(args.ext):
        images_ext = np.load(data_dir + "ext_images_256.npy")


    X_train   = single_sub_fmri[train_ind]
    Y_train   = images_single[train_ind]
    single_sub_train = single_sub[train_ind]

    X_val   = single_sub_fmri[val_ind]
    Y_val   = images_single[val_ind]
    single_sub_val = single_sub[val_ind]


    if args.v2c_mapping is not None:
        v2c_mapping = np.load(args.v2c_mapping)
    else:
        v2c_mapping = np.load(derived_data_dir + "v2c_128_mapping_gmm_v2.npy")


    print("load data")
    
    train_dataset = EmbedGraphDataset(X_train, Y_train,v2c_mapping , single_sub_train ,sub_num_voxels = num_voxels_subjects
                                         , sample = True ,num_voxels_to_sample = args.num_vox*1000, 
                                         num_centers = args.centers, transform=image_transform) 
    if(args.ext):
        ext_dataset = EmbedGraphDataset(fmri_ext, images_ext,v2c_mapping , single_sub_train,sub_num_voxels = num_voxels_subjects
                                       , sample = True ,num_voxels_to_sample = args.num_vox*1000,  rand_subject = True, 
                                       num_centers = args.centers, transform=image_transform) 
        full_dataset = DatasetExtWraper(train_dataset,ext_dataset, sample_factor = args.ext_sample_factor)
    
    else:
        full_dataset = train_dataset
    
    
    
    val_dataset   = EmbedGraphDataset(X_val, Y_val, v2c_mapping , single_sub_val  , sub_num_voxels = num_voxels_subjects
                                     , sample = False, num_centers = args.centers, transform=image_transform)
    
    
    custom_collate_fn = partial(collate, N_C=args.centers)
    
    train_dataloader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=2, #2
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=False
    )
    
    print(f"Created dataloaders - Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches", flush=True)
    
    
    #################################################     callbacks ###################################################
     
    
    
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(save_dir, name),   # folder to save in
    filename=name,         # name without extension
    save_last=True,          # save only the last epoch
    save_top_k=1,            # don't save intermediate/best
    monitor="val_loss",                     # metric to monitor
    mode="min"                              # 'min' for loss, 'max' for accuracy/AUC
)    
        

    logger = TensorBoardLogger(tensorbaord_dir, name=name)

    
    #################################################     load models ###################################################
    # Load models
    diffusion_engine = load_diffusion_engine(diffusion_model_cfg, diffusion_model_path)
            
    if(args.gnn_model_path == None):
        gnn_model = torch.load(save_dir+"decoder_clipg_ext-1_save.pth")
    else:
        gnn_model = torch.load(args.gnn_model_path)

    # Create combined model
    model = CombinedDiffusionEngine(diffusion_engine=diffusion_engine, gnn_model=gnn_model)
    model.learning_rate = LEARNING_RATE
    print("Created GnnDiffusionEngine with diffusion_engine and gnn_model", flush=True)
        
        
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu',
        devices=args.num_gpus,
        strategy="ddp",
        precision='16-mixed',
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        check_val_every_n_epoch=1,  # logs at every validation epoch
        enable_model_summary=False,
        logger=logger,
        callbacks=[checkpoint_callback],
       
    )
    
    # Start training
    print("Starting training...", flush=True)
    print("GPU memory before training:", flush=True)
    print_gpu_memory_usage()
    
    
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training complete!", flush=True)
    
    print("GPU memory after training:", flush=True)
    print_gpu_memory_usage()
    

if __name__ == "__main__":
    main()
