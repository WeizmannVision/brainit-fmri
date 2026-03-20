import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
import sys
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
from src.MindEyeV2.generative_models.sgm.util import (default, get_obj_from_str, instantiate_from_config)

from src.MindEyeV2.utils import unclip_recon_new

from omegaconf import OmegaConf
from src.MindEyeV2.generative_models.sgm.models.diffusion_no_lightning import DiffusionEngine_nolightning
import src.MindEyeV2.generative_models.sgm


class CombinedDiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        diffusion_engine = None,
        gnn_model= None,
        gnn_frozen=False,
        diffusion_frozen=False,
        save_vox=False,
        save_vox_dir=None,
    ):
        super().__init__()
        self.diffusion_engine = diffusion_engine
        self.gnn_model = gnn_model
        self.gnn_frozen = gnn_frozen
        self.diffusion_frozen = diffusion_frozen
        self.save_vox = save_vox
        self.save_vox_dir = save_vox_dir
    def init_from_ckpt(self, path: str) -> None:
        return self.diffusion_engine.init_from_ckpt(path)
    
    def state_dict(self):
        """Return state dict containing both models for Lightning checkpointing"""
        return {
            'diffusion_engine': self.diffusion_engine.state_dict(),
            'gnn_model': self.gnn_model.state_dict(),
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict for both models from Lightning checkpoint"""
        self.diffusion_engine.load_state_dict(state_dict['diffusion_engine'], strict=strict)
        self.gnn_model.load_state_dict(state_dict['gnn_model'], strict=strict)

    def save_models(self, save_path: str):
        """
        Save both diffusion_engine and gnn_model as complete objects
        
        Args:
            save_path: Path to save the combined model checkpoint
        """
        checkpoint = {
            "diffusion_engine": self.diffusion_engine,
            "gnn_model": self.gnn_model,
        }
        torch.save(checkpoint, save_path)
        print(f"Saved complete models to {save_path}")

    def load_models(self, save_path: str):
        """
        Load both diffusion_engine and gnn_model as complete objects
        
        Args:
            save_path: Path to the saved model checkpoint
        """
        checkpoint = torch.load(save_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.diffusion_engine = state_dict["diffusion_engine"]
        self.gnn_model = state_dict["gnn_model"]
        print(f"Loaded complete models from {save_path}")
        
        # Move to current device if needed
        if torch.cuda.is_available():
            self.diffusion_engine = self.diffusion_engine.cuda()
            self.gnn_model = self.gnn_model.cuda()

    def _init_first_stage(self, config):
        return self.diffusion_engine._init_first_stage(config)

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[1]

    def prepare_diff_batch(self, clip_embed,image_tensor):
        if clip_embed.dim() < 3:
            clip_embed = clip_embed.unsqueeze(0)        
        batch_size = len(clip_embed)
        original_size_val = torch.tensor([[256,256]] * batch_size) 
        crop_coords_val = torch.tensor([[0,0]] * batch_size)
        return  {
            "jpg": clip_embed.float(), 
            "input_img": image_tensor, 
            "original_size_as_tuple": original_size_val.to(image_tensor.device), 
            "crop_coords_top_left": crop_coords_val.to(image_tensor.device)
        }

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.diffusion_engine.decode_first_stage(z)
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.diffusion_engine.encode_first_stage(x)

    def forward(self, x, batch, already_computed_clip_embeds: bool = False):
        fmri, targets, indexes, edge_index_v2c = batch
        clip_embed = self.gnn_model.forward(fmri, indexes, edge_index_v2c)
        batch = self.prepare_diff_batch(clip_embed, x)
        return self.diffusion_engine.forward(x, batch, already_computed_clip_embeds)

    def training_step(self, batch, batch_idx):
        fmri, targets, indexes, edge_index_v2c = batch
        clip_embed = self.gnn_model.forward(fmri, indexes, edge_index_v2c)
        batch = self.prepare_diff_batch(clip_embed, targets)
        loss,loss_dict= self.diffusion_engine.training_step(batch, batch_idx)
        
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        if self.diffusion_engine.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=False, on_epoch=True
            )

        return loss
    
    def get_vector_suffix(self):
        batch={"jpg": torch.randn(1,3,1,1), # jpg doesnt get used, it's just a placeholder
        "original_size_as_tuple": torch.ones(1, 2) * 256,
        "crop_coords_top_left": torch.zeros(1, 2)}
        out = self.diffusion_engine.conditioner(batch)
        return out["vector"]
    
    def generate(self, fmri, indexes, edge_index_v2c, img_init = None,num_samples =1,start_step =14,num_steps =38):
        if(img_init is not None):
            img_init = torch.from_numpy(img_init)
            if img_init.dim() == 3: 
                img_init = img_init.unsqueeze(0)  
            img_init = img_init.permute(0,3,1,2)
            img_init = torch.nn.functional.interpolate(img_init, size=(256, 256), mode='bilinear').float() #/255
        else:
            start_step = 0 
            
        with torch.no_grad():
            clip_pred = self.gnn_model.forward(fmri, indexes, edge_index_v2c)
            if clip_pred.dim() == 2:
                clip_pred = clip_pred.unsqueeze(0)
            #print(clip_pred.shape)
            if(img_init is not None):
                z = self.diffusion_engine.encode_first_stage((img_init* 2 - 1).cuda())
            else:
                z = None
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16):   
            samples = unclip_recon_new(
                clip_pred, 
                self.diffusion_engine,
                vector_suffix = self.get_vector_suffix(),
                num_samples=num_samples,
                init_image_latent=z,
                start_step=start_step
            )
        return samples.cpu()
    
    def validation_step(self, batch, batch_idx):
        fmri, targets, indexes, edge_index = batch
        clip_embed = self.gnn_model.forward(fmri, indexes, edge_index)
        batch = self.prepare_diff_batch(clip_embed, targets)
        loss_dict = self.diffusion_engine.validation_step(batch, batch_idx)
        # Log validation metrics with val_ prefix
        val_dict = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            val_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        
        # Log only the first batch of validation images (5 samples) to wandb
        if batch_idx in [0,1]:
            with torch.no_grad():
                # Get sample images
                log_images = self.diffusion_engine.log_images(batch, N=5, sample=True)
                
                # Log to wandb - ensuring proper format for wandb
                for k, v in log_images.items():
                    if isinstance(v, torch.Tensor):
                        # Convert to numpy and proper range for wandb (0-255)
                        v_np = ((v.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                        print(f"v_np: {v_np.shape}")
                        # Log images to wandb
                        try:
                            import wandb
                            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
                                self.logger.experiment.log({
                                    f"val_images/{k}": [
                                        wandb.Image(img) for img in v_np
                                    ],
                                    "global_step": self.global_step
                                })
                        except (ImportError, AttributeError) as e:
                            print(f"Warning: Could not log images to wandb: {e}")
        return loss_dict

    def on_train_start(self, *args, **kwargs):
        return self.diffusion_engine.on_train_start(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        return self.diffusion_engine.on_train_batch_end(*args, **kwargs)

    def on_train_epoch_end(self):
        result = self.diffusion_engine.on_train_epoch_end()
        
        # Save voxel embeddings every 50 epochs if requested
        if self.save_vox and hasattr(self.gnn_model, 'voxel_embed') and self.current_epoch % 50 == 0:
            import os
            if self.save_vox_dir is None:
                save_dir = "Results/saved_models/voxel_embeds/"
            else:
                save_dir = self.save_vox_dir
            os.makedirs(save_dir, exist_ok=True)
            vox_path = os.path.join(save_dir, f"voxel_embed_epoch{self.current_epoch}.pt")
            torch.save(self.gnn_model.voxel_embed.data, vox_path)
            print(f"Saved voxel embeddings to {vox_path}", flush=True)
        
        return result

    @contextmanager
    def ema_scope(self, context=None):
        with self.diffusion_engine.ema_scope(context):
            yield

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return self.diffusion_engine.instantiate_optimizer_from_config(params, lr, cfg)

    def configure_optimizers(self):
        lr = self.learning_rate
        # Collect parameters from both diffusion_engine and gnn_model
        params: List[torch.nn.Parameter] = []
        
        # Add diffusion_engine model parameters
        if not self.diffusion_frozen:
            params.extend([
                p for p in self.diffusion_engine.model.parameters() if p.requires_grad
            ])
        
        # Add gnn_model parameters
        if not self.gnn_frozen:
            params.extend([
                p for p in self.gnn_model.parameters() if p.requires_grad
            ])

        # Add trainable parameters of the conditioner embedders
        for embedder in self.diffusion_engine.conditioner.embedders:
            if embedder.is_trainable:
                params.extend([p for p in embedder.parameters() if p.requires_grad])
        
        opt = self.diffusion_engine.instantiate_optimizer_from_config(params, lr, self.diffusion_engine.optimizer_config)
        if self.diffusion_engine.scheduler_config is not None:
            scheduler = instantiate_from_config(self.diffusion_engine.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

