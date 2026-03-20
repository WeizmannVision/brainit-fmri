import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
from src.DIP.models import get_net
from src.DIP.utils.common_utils import get_noise 
from torchvision import transforms, models

def fold_and_average(unfolded, output_size, kernel_size=2, stride=1):
    """
    Revert F.unfold(x, kernel_size, stride).transpose(1, 2), averaging overlaps.
    
    Args:
        unfolded: Tensor of shape (B, L, C * K * K)
        output_size: tuple (H, W), the height and width of the original image
        kernel_size: int or tuple
        stride: int or tuple
        
    Returns:
        Tensor of shape (B, C, H, W)
    """
    B, L, patch_dim = unfolded.shape
    K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    C = patch_dim // (K * K)

    # Transpose back to (B, C*K*K, L) for fold

    unfolded_t = unfolded.transpose(1, 2)

    # Fold to reconstruct the image
    folded = F.fold(unfolded_t, output_size=output_size, kernel_size=kernel_size, stride=stride)

    # Create normalization mask to average overlapping regions
    ones_input = torch.ones((B, C, *output_size), dtype=unfolded.dtype, device=unfolded.device)
    ones_unfolded = F.unfold(ones_input, kernel_size=kernel_size, stride=stride)
    ones_folded = F.fold(ones_unfolded, output_size=output_size, kernel_size=kernel_size, stride=stride)

    # Avoid divide-by-zero
    ones_folded[ones_folded == 0] = 1.0

    return folded / ones_folded



class LowLevelREC:
    
    def __init__(self, scales = 3):
        self.vgg_model = models.vgg16_bn(pretrained=True).features.eval().cuda()
        self.reg_noise_std = 1./30.
        self.LR = 0.001
        self.exp_weight=0.99
        self.input_depth = 32 
        self.num_iter = 1800 
        self.scales = scales

        
        
    def rec_image(self, target_features):
    
        dtype = torch.cuda.FloatTensor


        target_features = self.get_features_flat(target_features)
        net = get_net(self.input_depth, 'skip', 'reflection',
                    skip_n33d=128, 
                    skip_n33u=128,
                    skip_n11=self.scales-1,
                    num_scales=self.scales,
                    upsample_mode='bilinear').type(dtype)




        ## Optimize
        net_input = get_noise(self.input_depth, 'noise', (224, 224),var=0.1).type(dtype).detach()
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_avg = None
        last_net = None
        psrn_noisy_last = 0
        i = 0

        out = net(net_input)
        rec_img_np = out.detach().cpu().numpy()[0]


        # save tmp PSNR for different learning strategies
        def closure():

            nonlocal  i, out_avg, psrn_noisy_last, last_net, net_input
            if self.reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)
            out = net(net_input)

            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * self.exp_weight + out.detach() * (1 - self.exp_weight)
            out_resized  = F.interpolate(out, (112, 112),
                                    mode='bicubic', align_corners=False, antialias=True)   

            out_features = self.get_features_image(out_resized)


            loss = self.layer_mse_loss(out_features,target_features)
            loss.backward()

            out_np = out.detach().cpu().numpy()[0]
            out_avg_np = out_avg.detach().cpu().numpy()[0]


            i += 1
            return out_avg_np

        ## Optimizing 
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR)


        for j in range(self.num_iter):
            optimizer.zero_grad()
            img = closure()
            optimizer.step()

        torch.cuda.empty_cache()

        return img.transpose(1, 2, 0)
        
        
        
    def get_features_flat(self, feat):
        feat = torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat
        feat = feat.unsqueeze(0) if feat.ndim == 2 else feat
        feat = feat.float()
        feat_size = np.array([56**2,55**2,28**2,14**2,7**2])
        end = np.cumsum(feat_size)
        start= end-feat_size

        features = {}

        feat_1 = feat[:, start[0]:end[0]]
        feat_1 = (feat_1[:, :, :256] + feat_1[:, :, 256:]) / 2  
        feat_1 = fold_and_average(feat_1, output_size=(112, 112), kernel_size=2, stride=2)
        features['conv1'] = feat_1.cuda()

        feat_2 = feat[:, start[1]:end[1]]
        feat_2 = fold_and_average(feat_2, output_size=(56, 56), kernel_size=2, stride=1)
        features['conv2'] = feat_2.cuda()

        feat_3 = feat[:, start[2]:end[2]]
        feat_3 = (feat_3[:, :, :256] + feat_3[:, :, 256:]) / 2  
        feat_3 = feat_3.float().transpose(1, 2).reshape(-1, 256, 28, 28)
        features['conv3'] = feat_3.cuda()

        feat_4 = feat[:, start[3]:end[3]]
        feat_4 = feat_4.float().transpose(1, 2).reshape(-1, 512, 14, 14)
        features['conv4'] = feat_4.cuda()

        feat_5 = feat[:, start[4]:end[4]]
        feat_5 = feat_5.float().transpose(1, 2).reshape(-1, 512, 7, 7)
        features['conv5'] = feat_5.cuda()

        return features





    def get_features_image(self,image ,layers=None):
        
        mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,3,1,1).cuda()
        std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,3,1,1).cuda()
        image_in = (image-mean)/std

        if layers is None:
            layers = {'5': 'conv1',
                      '12': 'conv2', 
                      '22': 'conv3', 
                      '32': 'conv4',
                      '42': 'conv5'}

        features = {}
        x = image_in
        # model._modules is a dictionary holding each module in the model
        for name, layer in self.vgg_model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features



    def layer_mse_loss(self, features_pred, features_tar):
        """
        Args:
            features_pred (dict): Dictionary of predicted feature tensors
                                  (e.g., {'conv1': t1, 'conv2': t2, ...})
            features_tar (dict): Dictionary of target feature tensors
                                 (e.g., {'conv1': t1, 'conv2': t2, ...})
        """
        layer_losses = []

        # Iterate over the layers (assuming keys match)
        for key in features_pred.keys():
            # Get tensors
            pred_tensor = features_pred[key]
            # Move target tensor to the specified device
            tar_tensor = features_tar[key]

            # Calculate MSE for the current layer
            loss_i = F.mse_loss(pred_tensor, tar_tensor)
            layer_losses.append(loss_i)

        # Return the average loss across all layers
        return torch.stack(layer_losses).mean()