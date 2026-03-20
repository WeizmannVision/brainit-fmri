### adaapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            logit_scale = 1.0,
            reshape = False,
            filter_similar = False,
            threshold = 0.95,
            soft = False, 
            normalize = False,
            remove_mean = False,
            train_mean = None
        
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.logit_scale = logit_scale
        self.reshape = reshape
        self.filter_similar = filter_similar
        self.threshold = threshold
        self.soft = soft
        self.normalize = normalize
        self.remove_mean = remove_mean
        
        # Register train_mean as a buffer to make it untrainable
        if remove_mean:
            self.register_buffer('train_mean', train_mean)
   

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, text_features_neg = None):
        if(self.normalize):
            #image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)

            
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            if(text_features_neg is None):
                logits_per_image = logit_scale * image_features @ text_features.T
            else:
                text_features_exten = torch.cat([text_features,text_features_neg],0)
                logits_per_image = logit_scale * image_features @ text_features_exten.T
            
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, text_features_neg= None, output_dict=False):
        device = image_features.device
        # print(image_features.shape)
        # print(text_features.shape)

        image_features = image_features.squeeze()
        text_features = text_features.squeeze()

        if self.remove_mean:
            # Remove pre-computed training mean from both features
            image_features = image_features - self.train_mean.to(device)
            text_features = text_features - self.train_mean.to(device)

        if(text_features_neg is not None):
            text_features_neg = text_features_neg.squeeze()
        if(image_features.ndim == 3 ):
            if(self.reshape):
                image_features = image_features.reshape(-1, image_features.size(2))
                text_features = text_features.reshape(-1, text_features.size(2))
     
            else:
                image_features = image_features.view(-1, image_features.size(2))
                text_features = text_features.view(-1, text_features.size(2))
        
   
            
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, self.logit_scale)
        
        # if(self.filter_similar):
        #     logits_filter, _ = self.get_logits(text_features, text_features, self.logit_scale)
        #     logits_filter.detach().diagonal().zero_()
        #     logits_per_image = logits_per_image*(logits_filter<self.thershold)
        #     logits_per_text = logits_per_text*(logits_filter<self.thershold)

        if self.filter_similar:
            # text–text cosine*logit_scale similarities
            sim_tt, _ = self.get_logits(text_features, text_features, self.logit_scale)

            B   = sim_tt.size(0)
            eye = torch.eye(B, dtype=torch.bool, device=sim_tt.device)

            # drop pairs whose similarity is **above** the threshold (but keep the diagonal)
            mask_drop = (sim_tt >= self.threshold) & (~eye)   # False ⇢ keep, True ⇢ drop

            logits_per_image = logits_per_image.masked_fill(mask_drop, -float("inf"))
            logits_per_text  = logits_per_text .masked_fill(mask_drop, -float("inf"))
        
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        
        _, predicted = torch.max(logits_per_text, 1)
        acc = ((predicted==labels)*1.0).mean().item()
        
        
        if(self.soft):
            logits_target, _ = self.get_logits(text_features, text_features, self.logit_scale, text_features_neg = text_features_neg)
            
            pred_logprob_image = F.log_softmax(logits_per_image, dim=1)    # log q
            pred_logprob_text  = F.log_softmax(logits_per_text, dim=1)
            
            target_prob   = F.softmax(logits_target,   dim=1)    # p

            
            total_loss = (
            F.kl_div(pred_logprob_image, target_prob, reduction='batchmean') + F.kl_div(pred_logprob_text, target_prob, reduction='batchmean')
            ) / 2
            return total_loss, acc


            
            

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss, acc
    
    
class CombinedClipMSELoss(nn.Module):
    def __init__(self, clip_loss, mse_weight=0.75):
        super().__init__()
        self.clip_loss = clip_loss
        self.mse_weight = mse_weight
        
    def forward(self, image_features, text_features):
        clip_loss, acc = self.clip_loss(image_features, text_features)
        mse_loss = F.mse_loss(image_features, text_features)
        total_loss = (1 - self.mse_weight) * clip_loss + self.mse_weight * mse_loss
        return total_loss, acc