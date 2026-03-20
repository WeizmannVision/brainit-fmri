import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn as nn

import math
import numpy as np

import torch.nn.functional as F

class MapAttention(torch.nn.Module):
    def __init__(self, dim_query, dim_key, num_heads, l1 = 0, softmax = False):
        super(MapAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = dim_key
        self.project = torch.nn.Linear(dim_key, dim_query)
        self.l1 = l1
        self.sqrt_dim = np.sqrt(dim_key)
        self.softmax = softmax


    def forward(self, query, key, x, attn = None ):
        if(attn is None):
            query_proj = self.project(query)
            score = torch.tensordot(query_proj, key, dims=([2], [1]))
            if(self.softmax):
                #attn = F.softmax(score / self.sqrt_dim , -1)
                attn = F.softmax(score , -1)
            else:
                attn = score
        x = torch.bmm(attn,x)#B,d,C,heads,V #, dims=([1], [1])

        return x, attn
    
    
class MapAttention_dynamic(torch.nn.Module):
    def __init__(self, dim_query, dim_in, num_heads, l1 = 0, softmax = False):
        super(MapAttention_dynamic, self).__init__()
        self.num_heads = num_heads
        self.project = torch.nn.Linear(dim_query, dim_query)
        self.project_x = torch.nn.Linear(dim_in, dim_query)

        self.l1 = l1
        self.sqrt_dim = np.sqrt(dim_query)
        self.softmax = softmax


    def forward(self, query, x, attn = None ):
        if(attn is None):
            query_proj = self.project(query)
            key = self.project_x(x)
            score = torch.bmm(query_proj, key.transpose(1, 2))
           
            if(self.softmax):
                #attn = F.softmax(score / self.sqrt_dim , -1)
                attn = F.softmax(score , -1)
            else:
                attn = score
        x = torch.bmm(attn,x)#B,d,C,heads,V #, dims=([1], [1])

        return x, attn
 

class MapAttention_dynamic_pos(torch.nn.Module):
    def __init__(self, dim_query, dim_in, num_heads, l1 = 0, softmax = False):
        super(MapAttention_dynamic_pos, self).__init__()
        self.num_heads = num_heads
        self.project = torch.nn.Linear(dim_query, dim_query)
        self.project_x = torch.nn.Linear(dim_in, dim_query)


        self.l1 = l1
        self.sqrt_dim = np.sqrt(dim_query)
        self.softmax = softmax


    def forward(self, query, key, x, attn = None ):
        if(attn is None):
            query_proj = self.project(query)
            key = self.project_x(x)+key.unsqueeze(0)
            score = torch.bmm(query_proj, key.transpose(1, 2))
           
            if(self.softmax):
                #attn = F.softmax(score / self.sqrt_dim , -1)
                attn = F.softmax(score , -1)
            else:
                attn = score
        x = torch.bmm(attn,x)#B,d,C,heads,V #, dims=([1], [1])

        return x, attn
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




    



