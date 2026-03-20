import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch 
from sklearn.preprocessing import normalize
import numpy as np

TRANSFER_SUB = "transfer_sub"
centers_num = 128

model = torch.load(f"results/saved_models/encoder_{TRANSFER_SUB}.pth").eval()

vox_embed = model.voxel_embed.detach().cpu()
vox_embed_normalized = normalize(vox_embed, axis=1)

model = torch.load('results/saved_models/encoder_base.pth').eval()
vox_embed_ref = model.voxel_embed.detach().cpu()
vox_embed_ref_normalized = normalize(vox_embed_ref, axis=1)

file = np.load("data/derived_data/gmm_centers_128_v2.npz")

labels  = file["labels"]
centers = file["centers"]

with torch.no_grad():
    dis_l2 = torch.cdist(torch.from_numpy(vox_embed_ref_normalized),torch.from_numpy(vox_embed_normalized))

nn_vox = dis_l2.argmin(0)
nn_vox = nn_vox.numpy()
label_vox = labels[nn_vox]

mapping = np.ones([centers_num,(np.unique(label_vox,return_counts = True)[1]).max()],dtype = int)*-1

for c in range(centers_num):
    inds_c = np.where(label_vox==c)[0]
    mapping[c,:len(inds_c)] = inds_c

os.makedirs("data/transfer", exist_ok=True)
np.save(f"data/transfer/v2c_128_mapping_gmm_{TRANSFER_SUB}.npy", mapping)
np.save(f"data/transfer/v2c_{TRANSFER_SUB}_nnvox.npy", nn_vox)
