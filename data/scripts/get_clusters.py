import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch 
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

sys.path.append(os.getcwd())

# Load DINOv2 encoder
torch.hub.set_dir("data/external_models/torch_hub/")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

# Load encoder model
model = torch.load("results/saved_models/encoder_ch128.pth").eval()

vox_embed = model.voxel_embed.cpu().detach().numpy()
vox_embed.shape[0]

vox_embed_normalized = normalize(vox_embed, axis=1)

data_dir = 'data/nsd_data/'
num_voxels_subjects = np.load(data_dir + 'num_voxels_all_subjects.npy')
num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)
end_ind = np.cumsum(num_voxels_subjects)

start_ind = end_ind - num_voxels_subjects

centers_num = 128
gmm = GaussianMixture(n_components=centers_num, verbose=1)
gmm.fit(vox_embed_normalized)
labels = gmm.predict(vox_embed_normalized)
centers = gmm.means_

output_dir = 'data/derived_data/'
os.makedirs(output_dir, exist_ok=True)
np.savez(os.path.join(output_dir, "gmm_centers_" + str(centers_num) + ".npz"), labels=labels, centers=centers)

mapping = np.ones([centers_num, (np.unique(labels, return_counts=True)[1]).max()], dtype=int) * -1

for c in range(centers_num):
    inds_c = np.where(labels == c)[0]
    mapping[c, :len(inds_c)] = inds_c

np.save(os.path.join(output_dir, "v2c_" + str(centers_num) + "_mapping_gmm.npy"), mapping)
