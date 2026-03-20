import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as stat
log_interval = 100

def corr_percentiles(y,y_pred, per = [50,75,90]):
    num_voxels = y.shape[1]
    corr = np.zeros([num_voxels])

    for i in range(num_voxels):
        corr[i] = stat.pearsonr(y[:, i], y_pred[:, i])[0]
    corr = np.nan_to_num(corr)

    corr_per = []
    for p in per:
        corr_per.append(np.percentile(corr,p))
    return corr_per



def log_tensorboard(pred,target,epoch,mode,writer, NC = None):
    pred = np.concatenate(pred)
    target = np.concatenate(target)
    print(pred.shape, target.shape)

    mse = np.mean(np.square(pred-target))
    mae = np.mean(np.abs(pred-target))
    print(mode,mae,mse)

    writer.add_scalar(mode + '/mse' , mse, epoch)
    writer.add_scalar(mode + '/mae' , mae, epoch)

    per = [50,75,90]
    corr_per = corr_percentiles(pred,target, per = per)
    print(corr_per)
    
    for i,p in enumerate(per):
        writer.add_scalar(mode+'/corr_'+str(p), corr_per[i], epoch)
        
    
        
def train( model, device, train_generator, optimizer, epoch, writer, scheduler = None, temp_factor = None, use_features = False):
    model.train()
    #loss_average = 0
    #batch_num =0
    pred_arr = []
    target_arr = []

    for batch_idx, (data, target, vox_ind) in enumerate(train_generator):

        data, target, vox_ind = data.float().cuda(), target.float().cuda(), vox_ind.int().cuda()

        target_arr.append(target.cpu().numpy())
        optimizer.zero_grad()
        if(temp_factor is not None):
            temp = temp_factor**epoch
            predict = model(data, vox_ind,temp)
        else:
            if(use_features):
                predict = model.forward_features(data, vox_ind)
            else:
                predict = model(data, vox_ind)

        pred_arr.append(predict.cpu().detach().numpy())
        loss = F.mse_loss(predict, target)- 0.1*torch.mean(F.cosine_similarity(predict, target))
        #reg =  model.regularization()
        loss_total = loss #+ reg
        #print(loss_total)

        loss_total.backward()
        optimizer.step()
        if(scheduler != None):
            scheduler.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \treg: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_generator.dataset),
                       100. * batch_idx / len(train_generator), loss.item(),loss.item()))
    writer.add_scalar('lr' , np.log10(optimizer.param_groups[0]['lr']), epoch)


    log_tensorboard(target_arr,pred_arr,epoch,'train',writer)

def test(model, device, train_generator, test_generator,epoch,writer, NC = None, use_features = False):
    model.eval()
    #test_loss = 0
    pred_arr = []
    target_arr = []
    
    
    pred_arr = []
    target_arr = []
    
    
    with torch.no_grad():
        for data, target,vox_ind in test_generator:
            target_arr.append(target.cpu().numpy())

            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

            if(use_features):
                output = model.forward_features(data, vox_ind)
            else:
                output = model(data, vox_ind)
            pred_arr.append(output.cpu().detach().numpy())

    log_tensorboard(target_arr,pred_arr,epoch,'test',writer,NC)

    pred = np.concatenate(target_arr)
    target = np.concatenate(pred_arr)

    corr_per = corr_percentiles(pred,target, per = [75] )[0]
    return corr_per