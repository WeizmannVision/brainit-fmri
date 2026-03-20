import os
import sys
sys.path.append(os.getcwd())
gpu     = "0"#str(sys.argv[1])
layer_attn_temp_factor   = None#float(sys.argv[2])
r = 7#int(sys.argv[2])
alpha = 0.01#float(sys.argv[3])
model_size = 2 #int(sys.argv[4])

dropout     =  0.25  # float(sys.argv[4])
embed_dim_vox = 256#int(sys.argv[1])
inner_ch = 128 #int(sys.argv[1])


os.environ["CUDA_VISIBLE_DEVICES"] = gpu

select_layers= [1,6,12,18,24]
#select_layers= [1,5,10,15,20,24]   



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from torchvision import datasets, transforms


from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
device = torch.device("cuda")
import numpy as np


import scipy.stats as stat
import sklearn.model_selection as sk_m
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import copy
import sys
import math


import numpy as np
from utils.train_utils_enc import train
from models.encoder_models import Encoder, encoder_param
from utils.datasets import EncDataset
torch.set_default_tensor_type('torch.FloatTensor')

TRANSFER_SUB = "transfer_sub"
batch_size = 32
lr = 1e-3 #0.001
epochs = 30

#save_model = '/home/romanb/data/NSD_models/model.pth'
data_dir = "data/transfer/"

file = np.load(data_dir + f"{TRANSFER_SUB}_fmri.npz")
single_sub_fmri_train = file['train']

single_sub_train = np.zeros(single_sub_fmri_train.shape[0]).astype(int)

images = np.load(data_dir + f"{TRANSFER_SUB}_imgs_224.npz")
embeds_single_train = images['train']

num_voxels_subjects = np.array([single_sub_fmri_train.shape[1]])

NUM_VOXELS = int(num_voxels_subjects.sum())


layer_attn_temp = 20#layer_attn_temp
layer_attn_temp_factor = layer_attn_temp_factor#layer_attn_temp_factor

name = f"encoder_transfer_{TRANSFER_SUB}"

print(name)
enc_param = encoder_param(NUM_VOXELS)

enc_param.inner_ch = inner_ch
enc_param.drop_out = dropout

 #1280#
 #256#197
enc_param.embed_dim_vox = embed_dim_vox

dataloader_param = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 4}

#enc_param.init = 0.1
###################################################
from scipy.ndimage import shift

mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3]).astype(float)
std  = np.array([0.229, 0.224, 0.225]).reshape([1,1,3]).astype(float) 
def trans_imgs(imgs):
    imgs = imgs/255.0
    imgs = (imgs-mean)/std
    imgs =  imgs.transpose([2,0,1])
    imgs =  torch.from_numpy(imgs.astype(float)).float()
    return imgs


def rand_shift(img,max_shift = 0 ):
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted

def trans_imgs_shift(img, max_shift = 3):
    img = img/255.0
    img = rand_shift(img,max_shift)
    img = (img-mean)/std
    img =  img.transpose([2,0,1])
    img =  torch.from_numpy(img.astype(float)).float()
    return img


fmri_dataset_train = EncDataset(embeds_single_train, single_sub_fmri_train, single_sub_train, num_voxels_subjects, preprocess = trans_imgs_shift ,num_voxels_to_sample = 5000)
train_generator = data.DataLoader(fmri_dataset_train, **dataloader_param)



writer = SummaryWriter('logs/tensorboard/encoder_exp/'+name)  #
include_reg_tokens = False
if(include_reg_tokens):
    enc_param.in_spatial = 261
else:
    enc_param.in_spatial = 257

#select_layers= [1,3,6,9,12]

torch.hub.set_dir("data/external_models/torch_hub/")
if(model_size == 0):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    enc_param.in_channels = 384
if(model_size == 1):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    enc_param.in_channels = 768
elif(model_size == 2):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    enc_param.in_channels = 1024
    select_layers= [1,6,12,18,24]
    
elif(model_size == 3):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    enc_param.in_channels = 1536
    #select_layers= [1,10,20,30,40]
    select_layers= [1,5,10,15,20,25,30,35,40]

encoder = encoder.eval().cuda()   

## freeze weights
for param in encoder.parameters():
    param.requires_grad = False
    

    

def main():
    # Load pretrained model
    model = torch.load("results/saved_models/encoder_base.pth")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Reinitialize voxel_embed as trainable
    model.voxel_embed = torch.nn.Parameter((enc_param.init/(2*np.sqrt(enc_param.embed_dim_vox)))*torch.randn(NUM_VOXELS, enc_param.embed_dim_vox), requires_grad=True).float()
    
    model = model.cuda()
  
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad = True)
    
    for epoch in range(1,epochs+1):
        print(epoch)
        train( model, device, train_generator, optimizer, epoch, writer)
    
    # Save model after 30 epochs
    torch.save(model, "results/saved_models/"+name+".pth")
    print(f"Model saved after {epochs} epochs")
            



if __name__ == '__main__':
    main()