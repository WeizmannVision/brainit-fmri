
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
import math
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from models.layers_encoder import MapAttention, Mlp, MapAttention_dynamic_pos


class encoder_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.img_len = 224
        self.inner_ch = 32
        self.drop_out = 0.25
        self.embed_dim = 32 
        self.embed_dim_vox = 64 
        self.init = 0.1
        self.in_channels = 1280
        self.in_spatial = 257

class Encoder(nn.Module):
    def __init__(self,param, model, num_embeds = 3, select_layers = [12], r =3, alpha = 1, include_reg_tokens = False, num_reg_tokens = 4): 
        super().__init__()
        self.param = param 
        self.num_embeds = num_embeds
        self.num_layers = len(select_layers)
        self.total_num_layers = len(model.blocks)
        self.embed_model = model
        self.r = r
        self.lora_scale = alpha/r
        self.include_reg_tokens = include_reg_tokens
        self.num_reg_tokens = num_reg_tokens
        self.model_init()
        self.select_layers = select_layers
                
    def init_A(self,m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            #nn.init.xavier_normal_(m.weight, 1)
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
           
        
    def init_B(self,m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            nn.init.zeros_(m.weight)



    def model_init(self):
        param = self.param
        self.voxel_embed = nn.Parameter((param.init/(2*np.sqrt(param.embed_dim_vox)))*torch.randn(param.num_voxels, param.embed_dim_vox), requires_grad=True)
        self.layer_embed = nn.Parameter((1/(2*np.sqrt(param.embed_dim)))*torch.randn(param.embed_dim_vox,self.num_layers), requires_grad=True)
        self.feat_embed = nn.Parameter((1/(2*np.sqrt(param.embed_dim)))*torch.randn(param.embed_dim_vox,param.inner_ch*self.num_layers), requires_grad=True)
        self.pos_embed = nn.Parameter((1/(2*np.sqrt(param.embed_dim)))*torch.randn(param.in_spatial,param.embed_dim_vox), requires_grad=True)

        self.in_project = nn.ModuleList([nn.Linear(param.in_channels, param.inner_ch)   for i in range(self.num_layers)])
        self.norm_in = nn.ModuleList([nn.LayerNorm(param.in_channels)  for i in range(self.num_layers)])

        self.norm = nn.ModuleList([nn.LayerNorm(param.inner_ch)  for i in range(self.num_layers)])
        
        self.map_att =  nn.ModuleList([MapAttention_dynamic_pos(param.embed_dim_vox, param.inner_ch,1, softmax = True)  for i in range(self.num_layers)])
        self.mlp = nn.ModuleList([Mlp(in_features=param.inner_ch, hidden_features=param.inner_ch*4)  for i in range(self.num_layers)])

        self.DO = nn.ModuleList([nn.Dropout(p=param.drop_out)  for i in range(self.num_layers)])
        
        self.lora_A = nn.ModuleList([nn.Linear(self.r, param.in_channels, bias = False)   for i in range(self.total_num_layers)])
        self.lora_B = nn.ModuleList([nn.Linear(param.in_channels, self.r, bias = False)   for i in range(self.total_num_layers)])
        self.lora_A.apply(self.init_A)
        self.lora_B.apply(self.init_B)
        
    def forward(self, x, voxel_ind, voxel_embed = None):
        layer_i = 0
        attn = None
        X_cat = []
        if(voxel_embed is None):
            voxel_embed =  self.voxel_embed[voxel_ind.long()] 
        # -- patchify x
        x = self.embed_model.prepare_tokens_with_masks(x)


#         # -- fwd prop
        for i, blk in enumerate(self.embed_model.blocks):
            x = blk(x)
            x = x + self.lora_A[i](self.lora_B[i](x))*self.lora_scale


            if(i+1 in self.select_layers):
                if(self.include_reg_tokens):
                    x_i = self.norm_in[layer_i](x)
                else:
                    # Old (incorrect): x_i = self.norm_in[layer_i](x[:,:-self.num_reg_tokens])
                    # Register tokens are at positions 1:1+num_reg_tokens, so exclude them
                    x_i = self.norm_in[layer_i](torch.cat([x[:, :1], x[:, 1+self.num_reg_tokens:]], dim=1))

                x_i    = self.in_project[layer_i](x_i)
                x_i, _ = self.map_att[layer_i](voxel_embed,self.pos_embed, x_i)
                x_i    = self.norm[layer_i](x_i)
                x_i    = self.DO[layer_i](x_i)
                x_i    = x_i+self.mlp[layer_i](x_i)

                X_cat.append(x_i)
                layer_i+=1
       

        x = torch.cat(X_cat,dim =-1)
        W = torch.matmul(voxel_embed,self.feat_embed) 
        x = torch.mul(x,W)
        x = x.sum(-1)      
        return x
    
    
    def get_features(self, x):
        layer_i = 0 
        X_cat = []
        # -- patchify x
        x = self.embed_model.prepare_tokens_with_masks(x)


#         # -- fwd prop
        for i, blk in enumerate(self.embed_model.blocks):
            x = blk(x)
            x = x + self.lora_A[i](self.lora_B[i](x))*self.lora_scale


            if(i+1 in self.select_layers):
                if(self.include_reg_tokens):
                    x_i = self.norm_in[layer_i](x)
                else:
                    # Old (incorrect): x_i = self.norm_in[layer_i](x[:,:-self.num_reg_tokens])
                    # Register tokens are at positions 1:1+num_reg_tokens, so exclude them
                    x_i = self.norm_in[layer_i](torch.cat([x[:, :1], x[:, 1+self.num_reg_tokens:]], dim=1))

                x_i    = self.in_project[layer_i](x_i)
                X_cat.append(x_i)  
                layer_i+=1
        return torch.stack(X_cat,1)
    
    def forward_features(self, features, voxel_ind, voxel_embed = None):
        
        attn = None
      
        X_cat = []
        
        if(voxel_embed is None):
            voxel_embed =  self.voxel_embed[voxel_ind.long()] 
        for layer_i in range(features.shape[1]):
            x_i    = features[:,layer_i]        
            x_i, _ = self.map_att[layer_i](voxel_embed,self.pos_embed , x_i)
            x_i    = self.norm[layer_i](x_i)
            x_i    = self.DO[layer_i](x_i)
            x_i    = x_i+self.mlp[layer_i](x_i)
            X_cat.append(x_i)
            
        x = torch.cat(X_cat,dim =-1)
        W = torch.matmul(voxel_embed,self.feat_embed) 
        x = torch.mul(x,W)
        x = x.sum(-1)      
        return x
        
        
        

    def regularization(self):
        return 0#self.lc.regularization()             