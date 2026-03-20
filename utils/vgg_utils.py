"""
VGG-related utilities for feature extraction and multi-scale loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VGGFeat(nn.Module):
    """
    VGG Feature Extractor that extracts multi-scale features from VGG19-BN.
    
    Args:
        model: VGG model (typically VGG19-BN)
        layer_indices: List of layer indices to extract features from
        vgg_norm: Whether to normalize the extracted features
    """
    def __init__(self, model, layer_indices=[5,12,22,32,42], vgg_norm=False):
        super(VGGFeat, self).__init__()
        self.features = model.features[:layer_indices[-1]+1]
        self.layer_indices = layer_indices
        self.vgg_norm = vgg_norm

    def forward(self, x):
        with torch.no_grad():
            outputs = []
            out_i = 0 
            # Iterate over all layers and store outputs for selected indices
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if idx in self.layer_indices:

                    if(out_i == 0):
                        out = F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)
                        out = torch.cat([out,out], dim=-1)
                    elif(out_i == 1):
                        out = F.unfold(x, kernel_size=2, stride=1).transpose(1, 2)
                    else:
                        out = x.reshape(x.shape[0],x.shape[1],-1).transpose(1, 2)

                    if(out_i == 2):
                        out = torch.cat([out,out], dim=-1)

                    out_i+=1

                    if(self.vgg_norm):
                        out =  out/ out.norm(dim=-1, keepdim=True)
                    outputs.append(out)
                    
        return outputs
    

class ClipLoss_MS(nn.Module):
    """
    Multi-Scale CLIP Loss wrapper for VGG features.
    
    Splits predictions and targets into multiple scales and computes
    CLIP loss independently for each scale, then averages.
    
    Args:
        loss: Base loss function (typically ClipLoss)
        sample_target_vec: List of token counts per scale 
    """
    def __init__(self, loss, sample_target_vec=[512,512,128,64,16]):
        super().__init__()
        self.loss = loss
        sample_target_vec = np.array(sample_target_vec)
        self.end = np.cumsum(sample_target_vec)
        self.start = self.end - sample_target_vec
    
    def forward(self, image_features, text_features):
        loss = 0                
        acc = 0    
        scales = len(self.start)
        for i in range(scales):
            loss_i, acc_i = self.loss(
                image_features[:, self.start[i]:self.end[i]],
                text_features[:, self.start[i]:self.end[i]]
            )
            loss += loss_i          
            acc += acc_i  
        return loss / scales, acc / scales

