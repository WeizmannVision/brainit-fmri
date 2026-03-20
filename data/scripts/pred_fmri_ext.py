import sys
import os
import torch 
import imageio.v2 as iio
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
device = torch.device("cuda")


## this is only for model definition 
# encoder = torch.hub.load(
#     repo_or_dir   = 'data/external_models/dinov2',
#     model         = 'dinov2_vits14_reg',
#     pretrained    = False,
#     source        = 'local',
#     force_reload  = True,
#     trust_repo    = True
# )
torch.hub.set_dir("data/external_models/torch_hub/")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

encoder_model = torch.load('results/saved_models/encoder_ch128.pth').eval()
encoder_model =  encoder_model.to(device)


data_dir = 'data/nsd_data/'
ext_imgs = np.load(data_dir + "ext_images_224.npy")


import numpy as np
mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3]).astype(float)
std  = np.array([0.229, 0.224, 0.225]).reshape([1,1,3]).astype(float) 
def trans_imgs_shift(img):
    img = (img-mean)/std
    img =  img.transpose([2,0,1])
    img =  torch.from_numpy(img.astype(float)).float()
    return img


num_voxels_subjects = np.load(data_dir + 'num_voxels_all_subjects.npy')
num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)
NUM_VOXELS = int(num_voxels_subjects.sum())



embeds = np.zeros([ext_imgs.shape[0],NUM_VOXELS], dtype=np.float32)
intermediate_output = []
counter = 0
for i in range(ext_imgs.shape[0]):
    if(i%1000==0):
        print(i)
    image_tensor = trans_imgs_shift(ext_imgs[i]/255.0).unsqueeze(0)
    with torch.no_grad():
        pred = encoder_model(image_tensor.cuda(),torch.arange(NUM_VOXELS).unsqueeze(0))
    embeds[i] = pred.detach().cpu().numpy()
np.save("data/derived_data/ext_fmri.npy", embeds.astype(np.float16))
