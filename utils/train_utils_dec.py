import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as stat
#import wandb
import copy
import torch.distributed as dist
#from utils.gather import all_gather_with_grad

log_interval = 100

import time


 
def log_tensorboard(pred,target,corr,corr_diff,epoch,mode,writer, use_wandb = False):
    mse = np.mean(np.square(pred-target))
    mae = np.mean(np.abs(pred-target))
    if(writer is not None):
        writer.add_scalar(mode + '/mse' , mse, epoch)
        writer.add_scalar(mode + '/mae' , mae, epoch)

        writer.add_scalar(mode+'/corr', corr, epoch)
        writer.add_scalar(mode+'/corr_diff', corr_diff, epoch)
    if(use_wandb):
        wandb.log({"epoch": epoch, mode+"_mse": mse,  mode+"_mae": mae,  mode+"_corr": corr})

    print(mse,corr,corr_diff)

        
        
def train( model, device, train_generator, optimizer, epoch, writer, loss_func = F.mse_loss, use_wandb = False, loss_contrastive = False, feat_extractor = None, sample_target_vec = [512,512,128,64,16], metrics = None):
    model.train()
    acc_sum = 0
    loss_sum = 0
    count = 0 
    target_ids = None
    metrics_sum = {k: 0 for k in metrics.keys()} if metrics is not None else None
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()


    for batch_idx, (batch) in enumerate(train_generator):

        data, target, vox_x, edges_indexes_v2c = batch 
        data, target, vox_x, edges_indexes_v2c = data.float().to(device), target.float().to(device), vox_x.int().to(device),edges_indexes_v2c.int().to(device)
        
        if(feat_extractor is not None ): #and sample_target_vec is not None
            features = feat_extractor(target)
            target_list = []
            target_ids = []
            indices_shift = 0 
            for i, feat in enumerate(features):
                B,N,C = feat.shape
                if(sample_target_vec is not None):
                    batch_index = torch.arange(B).unsqueeze(1)
                    indices = torch.multinomial(torch.ones(B, N), num_samples=sample_target_vec[i], replacement=False)
                    
                    target_list.append(feat[batch_index,indices])
                    target_ids.append(indices+indices_shift)
                    indices_shift+=N
                else:
                    target_list.append(feat)
                    target_ids = None

            target = torch.cat(target_list,dim =1)
            if(sample_target_vec is not None):
                target_ids = torch.cat(target_ids,dim =1)
            
        # Mixed precision training
        with torch.cuda.amp.autocast():
            predict = model(data, vox_x, edges_indexes_v2c, out_tokens_inds = target_ids)
            
            if(loss_contrastive):
                loss, acc = loss_func(predict.squeeze(), target.squeeze())
            else:
                loss = loss_func(predict.squeeze(), target.squeeze())
                acc = 0
 

        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                metrics_sum[metric_name] += metric_fn(predict.squeeze(), target.squeeze()).cpu().detach().numpy()
        
        count += 1
        loss_sum += loss.cpu().detach().numpy()
        acc_sum += acc

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \treg: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item(),loss.item()))
    if(writer is not None):
        writer.add_scalar('lr' , np.log10(optimizer.param_groups[0]['lr']), epoch)
        writer.add_scalar('train/acc' , acc_sum/count, epoch)
        writer.add_scalar('train/clip_loss' , loss_sum/count, epoch)
    if(use_wandb):
        wandb.log({"epoch":epoch,"lr": optimizer.param_groups[0]['lr']})
        wandb.log({"epoch":epoch,"train_acc": acc_sum/count, "train_clip_loss": loss_sum/count})

            
        
    print("train", acc_sum/count, loss_sum/count)
        
    if metrics is not None:
        for metric_name, metric_sum in metrics_sum.items():
            metric_value = metric_sum / count
            if writer is not None:
                writer.add_scalar(f'train/{metric_name}', metric_value, epoch)
            if use_wandb:
                wandb.log({f"epoch": epoch, f"train_{metric_name}": metric_value})
            print(f"train {metric_name}: {metric_value}")
        


def test(model, device, test_generator,epoch,writer, loss_func = F.mse_loss, use_wandb = False, loss_contrastive = False, feat_extractor = None, sample_target_vec = [512,512,128,64,16], metrics = None, return_metric = None):
    model.eval()
    pred_arr = []
    target_arr = []
    count = 0
    acc_sum = 0
    loss_sum = 0
    target_ids = None
    metrics_sum = {k: 0 for k in metrics.keys()} if metrics is not None else None
    with torch.no_grad():
        for batch in test_generator:
            data, target, vox_x, edges_indexes_v2c = batch
            data, target, vox_x, edges_indexes_v2c = data.float().to(device), target.float().to(device), vox_x.int().to(device), edges_indexes_v2c.int().to(device)
            if(feat_extractor is not None):
                features = feat_extractor(target)
                target_list = []
                target_ids = []
                indices_shift = 0 
                for i, feat in enumerate(features):
                    B,N,C = feat.shape
                    batch_idx = torch.arange(B).unsqueeze(1)
                    indices = torch.multinomial(torch.ones(B, N), num_samples=sample_target_vec[i], replacement=False)

                    target_list.append(feat[batch_idx,indices])
                    target_ids.append(indices+indices_shift)
                    indices_shift+=N
                target = torch.cat(target_list,dim =1)
                target_ids = torch.cat(target_ids,dim =1)
          
            predict = model(data, vox_x,edges_indexes_v2c, out_tokens_inds = target_ids)

            if predict.ndim < target.ndim:
                predict = predict.unsqueeze(0)
           
            if(loss_contrastive):
                loss, acc = loss_func(predict.squeeze(), target.squeeze())
            else:
                loss = loss_func(predict.squeeze(), target.squeeze())
                acc = 0

            count+=1
            acc_sum+=acc
            loss_sum+=loss.cpu().detach().numpy()
                
            if metrics is not None:
                for metric_name, metric_fn in metrics.items():
                    metrics_sum[metric_name] += metric_fn(predict, target).cpu().detach().numpy()
                    
        if(writer is not None):
            writer.add_scalar('test/acc' , acc_sum/count, epoch)
            writer.add_scalar('test/clip_loss' , loss_sum/count, epoch)
        
        if(use_wandb):
            wandb.log({"epoch":epoch,"test_acc": acc_sum/count, "test_clip_loss": loss_sum/count})

    if metrics is not None:
        for metric_name, metric_sum in metrics_sum.items():
            metric_value = metric_sum / count
            if writer is not None:
                writer.add_scalar(f'test/{metric_name}', metric_value, epoch)
            if use_wandb:
                wandb.log({f"epoch": epoch, f"test_{metric_name}": metric_value})
            print(f"test {metric_name}: {metric_value}")
            
    print("test:", acc_sum/count, loss_sum/count)
    if(return_metric is not None):
        return metrics_sum[return_metric]/count
    else:
        return loss_sum/count



