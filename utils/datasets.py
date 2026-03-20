import torch
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
def to_torch(x):
    return torch.from_numpy(x)



class EncDataset(Dataset):
    def __init__(self, X, Y, sub, sub_num_voxels ,num_voxels_to_sample = int(5000), sample = True, preprocess = None):
        self.X = X
        self.Y = Y
        self.sub = sub
        self.sub_num_voxels = sub_num_voxels
        self.num_voxels_to_sample = num_voxels_to_sample
        self.sub_vox_start_ind = np.cumsum(sub_num_voxels)-sub_num_voxels
        self.sample = sample
        self.preprocess = preprocess
       
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if(self.preprocess is not None):
            x = self.preprocess(x)
        else:
            x = to_torch(x)
        y = self.Y[idx]
        sub = self.sub[idx]
        num_voxels = self.sub_num_voxels[sub]
        if(self.sample):
            vox_sel = np.random.randint(0,num_voxels,self.num_voxels_to_sample)
        else:
            vox_sel = np.arange(num_voxels)
        y = y[vox_sel]
        voxel_global_ind = vox_sel+self.sub_vox_start_ind[sub]
        return x , to_torch(y), to_torch(voxel_global_ind)


class EmbedGraphDataset(Dataset):
    def __init__(self, X, Y,v2c_mapping, sub  ,sub_num_voxels ,num_voxels_to_sample = int(5000), sample = True , num_centers = 256, rand_subject = False , transform = None):
        self.X = X#.astype(np.float32)
        self.Y = Y#.astype(np.float32)
        self.num_voxels_to_sample = num_voxels_to_sample
        self.sample = sample
        self.sub_num_voxels = sub_num_voxels
        self.v2c_mapping = v2c_mapping.astype(np.int32)
        self.sub = sub.astype(np.int32)
        self.num_centers = num_centers
        self.sub_vox_end_ind   = np.cumsum(sub_num_voxels).astype(np.int32)
        self.sub_vox_start_ind = (self.sub_vox_end_ind - sub_num_voxels).astype(np.int32)
        self.rand_subject = rand_subject
        self.num_subjects = len(sub_num_voxels)
        self.transform = transform
        
    def __len__(self):
        if isinstance(self.Y, list):
            return self.Y[0].shape[0]
        
        return self.Y.shape[0]
    
    def get_rand_item(self):
        idx = np.random.randint(self.__len__())
        return self.__getitem__(idx)
    
    
        
    def get_edges_indexes(self,NN ,vox_sel_s ,vox_sel_d):
        num_nn = NN.shape[1]
        NN = NN[vox_sel_d]
        vox_sel_d = vox_sel_d+vox_sel_s.max()+1

        all_voxels = np.concatenate([vox_sel_s,vox_sel_d])

        edge_s = NN.reshape(-1)
        edge_d = np.repeat(vox_sel_d,num_nn)
    
        select_edges = np.isin(edge_s,vox_sel_s)


        edge_s = edge_s[select_edges]
        edge_d = edge_d[select_edges]
        
        voxel_map = np.zeros([int(all_voxels.max()+1)],dtype = np.int32)
        voxel_map[all_voxels] = np.arange(len(all_voxels),dtype = np.int32)
        edge_s = voxel_map[edge_s]#+d_shift
        edge_d = voxel_map[edge_d]

        edges_indexes = np.stack([edge_s, edge_d],axis =0)
        return edges_indexes


   
    def __getitem__(self, idx, vox_sel = None):
        ## vox_sel - array voxels to select indexes go 0-#sub_sun_voxels for all subjects 
        y = (self.Y[idx]).astype(np.float32)
        x = (self.X[idx]).astype(np.float32)
        
        if(self.transform is not None):
             y = self.transform(y/255.0)

        if(self.rand_subject):
            sub = np.random.randint(self.num_subjects)
            x = x[self.sub_vox_start_ind[sub]:self.sub_vox_end_ind[sub]]
        else:
            sub = int(self.sub[idx])
        
        num_voxels = self.sub_num_voxels[sub]
        if(vox_sel is None):
            if(self.sample):
                vox_sel = np.random.randint(0,num_voxels,self.num_voxels_to_sample)
            else:
                vox_sel = np.arange(num_voxels)
        voxel_global_ind = vox_sel+self.sub_vox_start_ind[sub]
        
        x = x[vox_sel]
        
        centers = np.arange(self.num_centers)
        edges_indexes_v2c = self.get_edges_indexes(self.v2c_mapping, voxel_global_ind ,centers).astype(np.int32)
        
        if not torch.is_tensor(y):
            y = torch.from_numpy(y)
        
        return to_torch(x) , y, to_torch(voxel_global_ind), to_torch(edges_indexes_v2c)



class DatasetExtWraper(Dataset):
    def __init__(self, dataset, dataset_ext, sample_factor =2):
        self.dataset = dataset
        self.dataset_ext = dataset_ext
        self.sample_factor = sample_factor
        # Ensure both datasets have the same length, or handle accordingly
        

    def __len__(self):
        if self.sample_factor == -1:
            return len(self.dataset) + len(self.dataset_ext)
        return self.sample_factor*len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_factor == -1:
            # Sequential access: first dataset, then dataset_ext
            if idx < len(self.dataset):
                return self.dataset[idx]
            else:
                return self.dataset_ext[idx - len(self.dataset)]
        else:
            # Original behavior: randomly select which dataset to sample
            if (idx%self.sample_factor==0):
                return self.dataset[idx//self.sample_factor]  # Sample from the first dataset
            else:
                return self.dataset_ext.get_rand_item()  # Sample from the second dataset    
    


def collate(batch, N_C = 128):
    N = len(batch[0][2])
    x  = torch.stack([item[0] for item in batch],0)
    y  = torch.stack([item[1] for item in batch],0)
    voxel_x_ind = torch.stack([item[2] for item in batch],0)
    edges_indexes_v2c  = torch.cat([item[3]+i*(N+N_C) for i, item in enumerate(batch)],dim = 1)
    return [x,y,voxel_x_ind,edges_indexes_v2c]






