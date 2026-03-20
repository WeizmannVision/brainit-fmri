from omegaconf import OmegaConf
import sys
import os
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2','generative_models'))
from src.MindEyeV2.generative_models.sgm.models.diffusion import DiffusionEngine
from src.MindEyeV2.generative_models.sgm.models.diffusion_no_lightning import DiffusionEngine_nolightning
import src.MindEyeV2.generative_models.sgm
from generative_models.sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from torchvision import transforms
from tqdm import tqdm
from generative_models.sgm.util import append_dims
import numpy as np

# Directory constants
MINDEYE_MODEL_DIR = "data/external_models/MindEyeV2"
SDXL_MODEL_DIR = "data/external_models/SDXL"

def prep_sdxl(CONFIG_PATH=os.path.join(MINDEYE_MODEL_DIR, "unclip6.yaml"), 
              sdxl_config_path=os.path.join(SDXL_MODEL_DIR, "sd_xl_base.yaml"), 
              sdxl_ckpt_path=os.path.join(SDXL_MODEL_DIR, "zavychromaxl_v30.safetensors")):
    config = OmegaConf.load(CONFIG_PATH)
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config['params']['num_steps'] = 38
    config = OmegaConf.load(sdxl_config_path)
    config = OmegaConf.to_container(config, resolve=True)
    refiner_params = config["model"]["params"]

    network_config = refiner_params["network_config"]
    denoiser_config = refiner_params["denoiser_config"]
    first_stage_config = refiner_params["first_stage_config"]
    conditioner_config = refiner_params["conditioner_config"]
    scale_factor = refiner_params["scale_factor"]
    disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]
    
    base_engine = DiffusionEngine(network_config=network_config,
                        denoiser_config=denoiser_config,
                        first_stage_config=first_stage_config,
                        conditioner_config=conditioner_config,
                        sampler_config=sampler_config, # using the one defined by the unclip
                        scale_factor=scale_factor,
                        disable_first_stage_autocast=disable_first_stage_autocast,
                        ckpt_path=sdxl_ckpt_path)
    return base_engine,conditioner_config



def load_diffusion_engine(config_path=os.path.join(MINDEYE_MODEL_DIR, "unclip6.yaml"), 
                         ckpt_path=os.path.join(MINDEYE_MODEL_DIR, "unclip6_epoch0_step110000.ckpt")):
    """Load diffusion engine with configuration"""
    config = OmegaConf.load(config_path)
    print(f"loaded config from path: {config_path}", flush=True)
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    
    # Extract configuration parameters
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    loss_fn_config = unclip_params["loss_fn_config"]
    scheduler_config = unclip_params["scheduler_config"]
    no_cond_log = unclip_params["no_cond_log"]
    
    # Fix target for first_stage_config
    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    
    # Initialize DiffusionEngine
    model = DiffusionEngine_nolightning(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scheduler_config=scheduler_config,
        loss_fn_config=loss_fn_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
        no_cond_log=no_cond_log,
        input_key="input_img",
        use_already_computed_clip_embeds=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.model, 'enable_gradient_checkpointing'):
        model.model.enable_gradient_checkpointing()
    elif hasattr(model.model.diffusion_model, 'enable_gradient_checkpointing'):
        model.model.diffusion_model.enable_gradient_checkpointing()
    
    # Load pretrained weights
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        if "diffusion_engine" in state_dict:
            print(f"DiffusionEngine found in state_dict", flush=True)
            model.load_state_dict(state_dict['diffusion_engine'])
        else:
            model.load_state_dict(state_dict)
        print(f"DiffusionEngine loaded from checkpoint: {ckpt_path}", flush=True)
    
    return model




def enhance_recons(all_recons, 
                   unclip_config_path=os.path.join(MINDEYE_MODEL_DIR, "unclip6.yaml"), 
                   sdxl_config_path=os.path.join(SDXL_MODEL_DIR, "sd_xl_base.yaml"), 
                   sdxl_ckpt_path=os.path.join(SDXL_MODEL_DIR, "zavychromaxl_v30.safetensors"), 
                   num_samples=1):
    img_size = 768
    rec_imgs = []
    # seed all random functions
    
    base_engine,conditioner_config = prep_sdxl(unclip_config_path, sdxl_config_path, sdxl_ckpt_path)
    base_engine.eval().requires_grad_(False)
    base_engine.cuda()

    base_text_embedder1 = FrozenCLIPEmbedder(
        layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
        layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
    )
    base_text_embedder1.cuda()

    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
        version=conditioner_config['params']['emb_models'][1]['params']['version'],
        freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
        layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
        always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
        legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
    )
    base_text_embedder2.cuda()

    batch={"txt": "",
        "original_size_as_tuple": torch.ones(1, 2).cuda() * img_size, # 768 amit
        "crop_coords_top_left": torch.zeros(1, 2).cuda(),
        "target_size_as_tuple": torch.ones(1, 2).cuda() * 1024}
    out = base_engine.conditioner(batch)
    crossattn = out["crossattn"].cuda()
    vector_suffix = out["vector"][:,-1536:].cuda()
    print("crossattn", crossattn.shape)
    print("vector_suffix", vector_suffix.shape)
    print("---")
    batch_uc={"txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "original_size_as_tuple": torch.ones(1, 2).cuda() * img_size, # 768 amit
        "crop_coords_top_left": torch.zeros(1, 2).cuda(),
        "target_size_as_tuple": torch.ones(1, 2).cuda() * 1024}
    out = base_engine.conditioner(batch_uc)
    crossattn_uc = out["crossattn"].cuda()
    vector_uc = out["vector"].cuda()
    print("crossattn_uc", crossattn_uc.shape)
    print("vector_uc", vector_uc.shape)

    num_samples = 1 
    img2img_timepoint = 9 # higher number means more reliance on prompt, less reliance on matching the conditioning image
    base_engine.sampler.guider.scale = 5 
    def denoiser(x, sigma, c): return base_engine.denoiser(base_engine.model, x, sigma, c)

    # Convert numpy array to PyTorch tensor if needed
    if isinstance(all_recons, np.ndarray):
        all_recons = torch.from_numpy(all_recons).float()
        # If dimensions are (N, H, W, C), convert to (N, C, H, W)
        all_recons = all_recons.permute(0, 3, 1, 2)
    
    all_recons = transforms.Resize((img_size,img_size))(all_recons).float()
    all_enhancedrecons = None
    for img_idx in tqdm(range(len(all_recons))):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
            base_engine.sampler.num_steps = 25
            recon_image = all_recons[[img_idx]]           
            recon_image = recon_image.cuda()
            prompt = ""
            assert recon_image.shape[-1]==img_size
            z = base_engine.encode_first_stage(recon_image*2-1).repeat(num_samples,1,1,1) # image to image

            openai_clip_text = base_text_embedder1(prompt)
            clip_text_tokenized, clip_text_emb  = base_text_embedder2(prompt) 
            clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
            clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized),dim=-1)
            c = {"crossattn": clip_text_tokenized.repeat(num_samples,1,1), "vector": clip_text_emb.repeat(num_samples,1)}
            uc = {"crossattn": crossattn_uc.repeat(num_samples,1,1), "vector": vector_uc.repeat(num_samples,1)}

            noise = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).cuda()
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples,1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(init_z, cond=c, uc=uc, 
                                                                num_steps=base_engine.sampler.num_steps)
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(sigmas[:,timestep],
                                                            sigmas[:,timestep+1],
                                                            denoiser, noised_z, cond=c, uc=uc, gamma=0)
            samples_z_base = noised_z
            samples_x = base_engine.decode_first_stage(samples_z_base)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            samples = samples[0]
            # Convert from PyTorch (C, H, W) to numpy (H, W, C) format
            samples = samples.permute(1, 2, 0).cpu().numpy()
            rec_imgs.append(samples)
    return np.stack(rec_imgs)