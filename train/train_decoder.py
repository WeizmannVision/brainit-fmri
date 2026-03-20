import os
import sys
#base_dir = "" #/home/romanb/FmriVision/
sys.path.append(os.getcwd())


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
print("start")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from models.decoder_models import dec_param, Decoder

import numpy as np

from sklearn.model_selection import train_test_split
from utils.datasets import EmbedGraphDataset, collate, DatasetExtWraper
from utils.train_utils_dec import train, test
from utils.clip_loss import ClipLoss
from utils.vgg_utils import VGGFeat, ClipLoss_MS
from tensorboardX import SummaryWriter
from torch_geometric.nn.conv import SAGEConv,GCNConv, GATConv, TransformerConv, SimpleConv
import argparse
from functools import partial
from torchvision import transforms, models

try:
    import torchmetrics
except ImportError:
    print("torchmetricsnot available")


#################################################   parse arguments ###################################################
parser = argparse.ArgumentParser()
parser.add_argument('--CLIPG', dest='clipg', action='store_const', const=True, default=False, help='CLIP-guided objective')
parser.add_argument('--CONT', dest='contrastive', action='store_const', const=True, default=False, help='contrastive loss')
parser.add_argument('--SAVE', dest='save', action='store_const', const=True, default=False, help='save model')
parser.add_argument('--C', dest='centers', type=int, default=128, help='number of centers')
parser.add_argument('--NUM_VOX', dest='num_vox', type=int, default=15, help='number of voxels')
parser.add_argument('--DIM', dest='dim', type=int, default=512, help='dimension of inner representation')
parser.add_argument('--HEADS', dest='heads', type=int, default=8, help='NUmber of transformers heads')
parser.add_argument('--BLOCKS', dest='num_blocks', type=int, default=5, help='NUmber of Transformers blocks')
parser.add_argument('--BATCH', dest='batch', type=int, default=128, help='dimension of inner representation')
parser.add_argument('--TEST_BATCH', dest='test_batch', type=int, default=16, help='test batch size')
parser.add_argument('--LR', dest='lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--PATIENCE', dest='patience', type=int, default=5, help='patience for learning rate scheduler')
parser.add_argument('--ADAMW', dest='adamw', action='store_const', const=True, default=True, help='USE Adamw')
parser.add_argument('--WARM', dest='warmup', action='store_const', const=True, default=True, help='warmup')
parser.add_argument('--WARMUP_EPOCHS', dest='warmup_epochs', type=int, default=15, help='number of warmup epochs')
parser.add_argument('--EXT', dest='ext', action='store_const', const=True, default=False, help='include external data')
parser.add_argument('--SAMPLE', dest='ext_sample_factor', type=int, default=-1, help='external freq sample')
parser.add_argument('--IMAGENT', dest='imagnet_ext', action='store_const', const=True, default=False, help='use imagenet')
parser.add_argument('--XFORMERS', dest='xformers', action='store_const', const=True, default=True, help='use xformers attention')
parser.add_argument('--FINETUNE', dest='finetune', type=str, default=None, help='finetune model path')
parser.add_argument('--V2C_MAPPING', dest='v2c_mapping', type=str, default=None, help='path to v2c mapping file')
parser.add_argument('--SAVE_LAST', dest='save_last', action='store_const', const=True, default=False, help='save only last model instead of best')
parser.add_argument('--VGG', dest='vgg', action='store_const', const=True, default=False, help='use VGG feature extraction mode')

args = parser.parse_args()

# Adjust batch size for VGG mode (cap at 64 if larger)
if args.vgg and args.batch > 64:
    args.batch = 64

#################################################   config ###################################################


use_wandb = False
name = "decoder"
if(args.centers!=128):
    name+="_c"+str(args.centers)

if(args.vgg):
    name+="_vgg"

if(args.contrastive):
    name+="_cont"   

if(args.clipg):
    name+="_clipg"    
    
      
if(args.ext):
    name+="_ext"+str(args.ext_sample_factor)

if(args.imagnet_ext):
    name+="_imagnet"

if(args.dim!=512):
    name+="_dim"+str(args.dim)
    
if(args.num_vox!=15):
    name+="_numvox_"+str(args.num_vox)

    
if(args.heads!=8):
    name+="_heads"+str(args.heads)    
    
if(args.num_blocks!=5):
    name+="_blocks"+str(args.num_blocks)   

if(args.batch!=128):
    name+="_batch"+str(args.batch)

if(args.lr!=5e-4):  
    name+="_lr"+str(args.lr)
    
if(args.warmup_epochs!=15):
    name+="_warmup_epochs"+str(args.warmup_epochs)     
   
           
if(use_wandb):
    wandb.init(
        # set the wandb project where this run will be logged
        project="gnn_pred_arr",
    )
    wandb.config.update(vars(args))

if(args.save):
    name+="_save"

if(args.finetune is not None):
    name+="_finetune"

data_dir = "data/nsd_data/"
derived_data_dir = "data/derived_data/"
save_dir = "results/saved_models/"
tensorbaord_dir =  'logs/tensorboard/decoder/'
os.makedirs(tensorbaord_dir, exist_ok=True)

conv_layer = TransformerConv #GCNConv #TransformerConv_edge

embed_dim_vox = args.dim
print(name)
writer = SummaryWriter(tensorbaord_dir+name)

num_voxels_subjects = np.load(data_dir + 'num_voxels_all_subjects.npy')
num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)
N = num_voxels_subjects.sum()
    
param = dec_param(N)
param.heads = args.heads
param.embed_dim_vox = embed_dim_vox
param.num_centers = args.centers
param.proj_mlp = False
param.num_blocks = args.num_blocks
param.xformers = args.xformers
param.out_dim = 1024
param.locally_connected = False


if(args.vgg):
    param.out_tokens = 56**2+55**2+28**2+14**2+7**2  # Multi-scale VGG output: 2029
    param.out_dim = 512

if(args.clipg):
    param.out_tokens = 256
    param.out_dim = 1664

lr = args.lr
epochs = 75
device = torch.device("cuda")
#################################################     load data ###################################################


  

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

if(args.vgg):
    embed = np.load(data_dir + "all_images_v2_112.npy")
    embed = embed[type_sample == 1]
    
    if(args.ext):
        embed_ext = np.load(data_dir + "ext_images_112.npy")

if(args.clipg):
    train_total_acc = False
    embed = np.load(data_dir + "all_images_v2_clip.npy")
    embed = embed[type_sample == 1]

    if(args.ext):
        embed_ext = np.load(data_dir + "ext_images_clip.npy")


X_train   = single_sub_fmri[train_ind]
Y_train   = embed[train_ind]
single_sub_train = single_sub[train_ind]

X_val   = single_sub_fmri[val_ind]
Y_val   = embed[val_ind]
single_sub_val = single_sub[val_ind]


if args.v2c_mapping is not None:
    v2c_mapping = np.load(args.v2c_mapping)
else:
    v2c_mapping = np.load(derived_data_dir + "v2c_128_mapping_gmm_v2.npy")


print("load data")
#################################################  data generators ###################################################

# VGG preprocessing
if args.vgg:
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
else:
    preprocess = None

dataloader_param = {'batch_size': args.batch,
          'shuffle': True,
          'num_workers': 4}


dataloader_test_param = {'batch_size': args.test_batch,
          'shuffle': False,
          'num_workers': 4}

train_loader = EmbedGraphDataset(X_train, Y_train,v2c_mapping , single_sub_train ,sub_num_voxels = num_voxels_subjects, sample = True ,num_voxels_to_sample = args.num_vox*1000, num_centers = args.centers, transform=preprocess) #
if(args.ext):
    ext_loader = EmbedGraphDataset(fmri_ext, embed_ext,v2c_mapping , single_sub_train,sub_num_voxels = num_voxels_subjects, sample = True ,num_voxels_to_sample = args.num_vox*1000,  rand_subject = True, num_centers = args.centers, transform=preprocess) #
    
    full_loader = DatasetExtWraper(train_loader,ext_loader, sample_factor = args.ext_sample_factor)
    
val_loader   = EmbedGraphDataset(X_val, Y_val, v2c_mapping , single_sub_val  , sub_num_voxels = num_voxels_subjects, sample = False, num_centers = args.centers, transform=preprocess )

custom_collate = partial(collate, N_C=args.centers)


if(args.ext):
    train_generator = DataLoader(full_loader, **dataloader_param,collate_fn=custom_collate) #
else:
    train_generator = DataLoader(train_loader, **dataloader_param,collate_fn=custom_collate) #

val_generator = DataLoader(val_loader, **dataloader_test_param,collate_fn=custom_collate) #

print("data loaders")

#################################################  Train ###################################################

warmup_epochs = args.warmup_epochs


def warmup(epoch, warmup_epochs = warmup_epochs, start_lr=1e-3 ):
    factor = 1
    if(epoch<warmup_epochs):
        factor =  start_lr+(1-start_lr)* ((float(epoch) / float(warmup_epochs)) ** 3)
    return factor



def main():
    model = Decoder(param, conv_layer = conv_layer) #, vox2centers_bitmap,centers2vox_list
    if(args.finetune is not None):
        model = torch.load(args.finetune)
        # Freeze all parameters
        for model_param in model.parameters():
            model_param.requires_grad = False
        # Only unfreeze voxel_embed parameter
        model.voxel_embed.requires_grad = True
    model = model.cuda()
    model.float()
    
    # Initialize VGG feature extractor if VGG mode is active
    vgg_feat = None
    if args.vgg:
        vgg_bn = models.vgg16_bn(pretrained=True).eval()
        for param_vgg in vgg_bn.parameters():
            param_vgg.requires_grad = False
        vgg_feat = VGGFeat(vgg_bn).float().cuda()
    
    print("model init")

    if(args.adamw):
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad = True)
        
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True,mode='min', factor=0.1,threshold=0.001,threshold_mode ='abs')
    if(args.warmup):
        scheduler_warmup = LambdaLR(optimizer, warmup) 
    best_metric = 1e5
    metric = 0
    metrics_dict={}
    metrics_dict['mse'] = lambda pred, target: F.mse_loss(pred, target)
    metrics_dict['mae'] = lambda pred, target: F.l1_loss(pred, target)
    return_metric = None
    
    if(args.contrastive):
        if args.vgg:
            # Use multi-scale CLIP loss for VGG
            loss_func_ = ClipLoss(reshape=True)
            loss_func = ClipLoss_MS(loss_func_)
        else:
            loss_func = ClipLoss()
    else:
        loss_func = F.mse_loss
        return_metric = 'mse'
    
    metric = test(model, device, val_generator, 0, writer, loss_func=loss_func, 
                 metrics=metrics_dict, return_metric=return_metric, feat_extractor=vgg_feat,
                 loss_contrastive=args.contrastive)

    for epoch in range(1,epochs+1):
        print(epoch)        
        if(args.warmup and epoch<warmup_epochs+1):
            scheduler_warmup.step()
        else:
            scheduler.step(metric)  # loss
            
        print("LR",optimizer.param_groups[0]['lr'])
        
        train(model, device, train_generator, optimizer, epoch, writer, loss_func=loss_func, 
             metrics=metrics_dict, feat_extractor=vgg_feat, loss_contrastive=args.contrastive)
       
        metric = test(model, device, val_generator, epoch, writer, loss_func=loss_func, 
                     metrics=metrics_dict, return_metric=return_metric, feat_extractor=vgg_feat,
                     loss_contrastive=args.contrastive)
        print("metric sch:",metric)
        
        if not args.save_last:
            if(metric<best_metric):
                best_metric = metric
                if(args.save):
                    torch.save(model, save_dir+str(name)+".pth")
                    print("save best",epoch,best_metric)
        if(optimizer.param_groups[0]['lr']<1e-7):
            break       
    
    # Save last model if save_last flag is set
    if args.save_last and args.save:
        torch.save(model, save_dir+str(name)+".pth")
        print("save last model")

if __name__ == '__main__':
    main()



