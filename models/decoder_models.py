import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import  TransformerConv
import numpy as np
from models.layers_transformer import Mlp, Block, Attention, CrossAttention, BlockCross, Block_xformer, BlockCross_xformer

class dec_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.embed_dim_vox = 1024 
        self.init =  0.1
        self.out_dim = 512
        self.num_centers = 256
        self.heads = 8
        self.out_tokens = 1 
        self.num_blocks = 2
        self.xformers = True



def _expand_token(token, batch_size):
    return token.view(1, token.shape[0], token.shape[1]).expand(batch_size, -1, -1) 


    
class Decoder(nn.Module):
    def __init__(self, param, conv_layer = TransformerConv): 
        super().__init__()
        self.param = param
        self.conv_layer = conv_layer
        self.model_init()
    def model_init(self):
        param = self.param
        self.voxel_embed = nn.Parameter((param.init/(2*np.sqrt(param.embed_dim_vox)))*torch.randn(param.num_voxels, param.embed_dim_vox), requires_grad=True)
        centers_dim = param.embed_dim_vox
        self.centers = nn.Parameter((param.init/(2*np.sqrt(centers_dim)))*torch.randn(param.num_centers, centers_dim, requires_grad=True))
            
        self.pred_tokens = nn.Parameter((param.init/(2*np.sqrt(param.embed_dim_vox)))*torch.randn(param.out_tokens, param.embed_dim_vox), requires_grad=True)
        self.conv_centers = self.conv_layer(centers_dim,centers_dim)

        self.relu  = torch.nn.ReLU()
        self.norm_centers =   nn.LayerNorm(param.embed_dim_vox)
        

        self.proj = nn.Linear(param.embed_dim_vox, param.out_dim)
        
        if(param.xformers):
            self.blocks = nn.ModuleList([
                Block_xformer(param.embed_dim_vox , num_heads=param.heads)
                for _ in range(param.num_blocks)])


            self.block_crosses = nn.ModuleList([
                BlockCross_xformer(dim=param.embed_dim_vox, num_heads=param.heads)
                for _ in range(param.num_blocks + 1)
            ])
            
        else:
            self.blocks = nn.ModuleList([
                Block(param.embed_dim_vox, num_heads=param.heads)
                for _ in range(param.num_blocks)])


            self.block_crosses = nn.ModuleList([
                BlockCross(dim=param.embed_dim_vox, num_heads=param.heads)
                for _ in range(param.num_blocks + 1)
            ])


    def forward(self, x, voxel_x ,v2c_mapping, out_tokens_inds = None, get_attn = False):
        param = self.param
        edge_index_v2c = v2c_mapping
        voxel_embed =  self.voxel_embed[voxel_x.long()]
        x = x.unsqueeze(-1)*voxel_embed
        B,N,C = x.shape
        Nc = N+param.num_centers

        x = x.reshape(B,N,C)
        centers = _expand_token(self.centers,B)
        x = torch.cat([x,centers[:,:,:C]],1)
        x = x.reshape(B*Nc,C)
        x = self.conv_centers(x,edge_index = edge_index_v2c.long())
        x = x.reshape(B,Nc,C)#[:,-num_query:]
        x = x[:,-param.num_centers:]
        x = self.relu(x)
        x = self.norm_centers(x)
        
        if(out_tokens_inds is not None):
            pred_tokens = self.pred_tokens[out_tokens_inds]
        else:
            pred_tokens = _expand_token(self.pred_tokens,B)
        
        if get_attn:
            attn_weights = []
        
        x_pred_update, attn = self.block_crosses[0](x, pred_tokens)
        x_pred = x_pred_update + pred_tokens
        if get_attn:
            attn_weights.append(attn)

        # Iterate over the number of main blocks.
        for i in range(self.param.num_blocks):
            # Process x using the i-th main block.
            x = self.blocks[i](x)
            # Update predictions with the corresponding cross block.
            # The cross block index is i+1 because we already used index 0.
            x_pred_update, attn = self.block_crosses[i+1](x, x_pred)
            x_pred = x_pred + x_pred_update
            if get_attn:
                attn_weights.append(attn)
        
        x_pred = self.proj(x_pred).squeeze()
        
        if get_attn:
            # Stack all attention weights along dim 0
            attn_stacked = torch.stack(attn_weights, dim=0)
            return x_pred, attn_stacked
        return x_pred





