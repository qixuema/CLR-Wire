# sample.py

import argparse
import torch
import json
from torch import is_tensor
import os
import sys
import numpy as np
from einops import rearrange, repeat

import logging
from torch.utils.data import DataLoader

# Add the parent directory of this file to Python’s module search path,
# allowing imports from modules located in the project’s root directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from model_loader import load_wireset_vae, load_curve_vae
from utils import denorm_curves, polylines_to_png

from src.utils.config import load_config, NestedDictToClass
from src.utils.helpers import setup_logging, cycle
from src.dataset.dataset_fn import curve_to_mean_custom_collate
from src.dataset.dataset import WireframeNormDataset



logger = logging.getLogger(__name__)
setup_logging()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_curves(
    curves: np.ndarray,
    uid: str,
    tgt_dir_path: str,
    save_png: bool = False,
):
    tgt_npy_dir = tgt_dir_path + '/npy'
    os.makedirs(tgt_npy_dir, exist_ok=True)
    tgt_file_path = tgt_npy_dir + f'/{uid}.npy'
    np.save(tgt_file_path, curves)
    
    if save_png:
        tgt_png_dir = tgt_dir_path + '/png'
        os.makedirs(tgt_png_dir, exist_ok=True)
        tgt_png_file_path = tgt_png_dir + f'/{uid}.png'
        polylines_to_png(curves, filename=tgt_png_file_path)


def curve_recon(
    curve_vae_cfg: NestedDictToClass, 
    curve_model_type: str = 'ae',
    device: str = 'cuda:0'
):
    
    logger.info("load Curve VAE model...")

    # Model loading
    curve_model = load_curve_vae(curve_vae_cfg, curve_model_type, device)
    
    # Ensure target directory exists
    os.makedirs(curve_vae_cfg.data.recon_dir_path, exist_ok=True)

    # load dataset
    logger.info("load dataset...")
    dataset = WireframeNormDataset(
        dataset_path=curve_vae_cfg.data.recon_wireframe_norm_file_path,
        correct_norm_curves=True,
    )

    # Initialize dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=10 if curve_vae_cfg.batch_size > 10 else curve_vae_cfg.batch_size, # use smaller of 10 or batch_size as num_workers
        pin_memory=True,
        drop_last=False,
        collate_fn=curve_to_mean_custom_collate,
    )
    
    dl_iter = dataloader
    logger.info("start reconstruction...")
    logger.info(f"num_iter: {len(dataloader)}")

    for batch_idx, data in enumerate(dl_iter):
        uids, num_curves, vertices_list, adjs_list, norm_curves = (
            data['uid'], data['num_curves'], data['vertices'], data['adjs'], data['norm_curves']
        )

        # Accumulate indices to partition the dataset
        accumulative_indices = np.cumsum(num_curves)
        accumulative_indices = np.insert(accumulative_indices, 0, 0)   

        # move data to appropriate device
        data = norm_curves.to(device)
        bs = data.shape[0]
        
        sample_points_num = curve_vae_cfg.model.sample_points_num
        t = torch.linspace(0, 1, sample_points_num, device=device)
        t = repeat(t, 't -> b t', b=bs)

        # Perform reconstruction
        with torch.no_grad():
            output = curve_model(
                data = data,
                t = t,
                sample_posterior = False,
                return_loss = False,
                return_std = (curve_model_type == 'en'),
            )

            if curve_model_type != 'en':
                output = output.sample

        if curve_model_type == 'ae':        
            output = rearrange(output, 'bs c n -> bs n c')

            norm_curves = output.detach().cpu().numpy()

            # Fix boundary points
            norm_curves[:, 0, :] = np.array([-1, 0, 0])
            norm_curves[:, -1, :] = np.array([1, 0, 0])
                    
            # Process each sample
            for i in range(len(num_curves)):
                start_idx = accumulative_indices[i]
                end_idx = accumulative_indices[i + 1]
                
                norm_curves_i = norm_curves[start_idx:end_idx]

                vertices_i = vertices_list[i]
                adjs_i = adjs_list[i]
                segments_i = vertices_i[adjs_i]
                
                uid = uids[i]

                # Denormalize the edges points
                curves = denorm_curves(norm_curves_i, segments_i)
            
                # Save the reconstructed edges points to a file
                save_curves(curves, uid, curve_vae_cfg.data.recon_dir_path, save_png=True)
    
        elif curve_model_type == 'en':
            mu, std = output
            
            mu = rearrange(mu, 'bs c n -> bs n c')
            std = rearrange(std, 'bs c n -> bs n c')
            
            mu = mu.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            
            for i in range(len(num_curves)):
                start_idx = accumulative_indices[i]
                end_idx = accumulative_indices[i + 1]
                uid = uids[i]
                mu_i = mu[start_idx:end_idx]
                std_i = std[start_idx:end_idx]

                tgt_file_path = curve_vae_cfg.data.recon_dir_path + f'/{uid}.npy'
                
                zs = np.concatenate([mu_i, std_i], axis=-1)
                
                np.save(tgt_file_path, zs)


        logger.info(f"batch {batch_idx} done.")

    exit()

def main(
    curve_vae_cfg: NestedDictToClass, 
    wireframe_vae_cfg: NestedDictToClass, 
    device_idx: int=0,
    is_curve_recon: bool=True,
    curve_model_type: str = 'ae',
    wireframe_model_type: str = 'en',
):
    
    isDebug = True if sys.gettrace() else False    
    device = f'cuda:{device_idx}' if torch.cuda.is_available() else "cpu"
    set_seed(3407)

    if isDebug:
        curve_vae_cfg.batch_size = 1
        wireframe_vae_cfg.use_wandb_tracking = False
        wireframe_vae_cfg.batch_size = 2048

    if is_curve_recon:
        curve_recon(
            curve_vae_cfg, 
            curve_model_type=curve_model_type,
            device=device,
        )
    else:
        wireset_recon(
            wireframe_vae_cfg, 
            curve_vae_cfg, 
            wireframe_model_type=wireframe_model_type,
            device=device, 
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE Reconstruction")
    parser.add_argument("-r","--is_curve_recon", type=int, required=False, help="Is curve recon", default=1)
    parser.add_argument("-c","--curve_vae_config", type=str, required=False, help="Path to curve VAE config file.",
                        default="src/configs/train_curve_vae.yaml")
    parser.add_argument("-w","--wireframe_vae_config", type=str, required=False, help="Path to wireframe VAE config file.",
                        default="src/configs/train_wireframe_vae.yaml")
    parser.add_argument("-ct","--curve_model_type", type=str, required=False, help="Curve model type", default="ae")
    parser.add_argument("-wt","--wireframe_model_type", type=str, required=False, help="Wireframe model type", default="ae")
    
    args, unknown = parser.parse_known_args()  # unknown contains any extra arguments    
     
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")
    
    return args


if __name__ == "__main__":
    vae_cfg = parse_args()
    curve_vae_cfg = load_config(vae_cfg.curve_vae_config)
    wireset_vae_cfg = load_config(vae_cfg.wireframe_vae_config)

    curve_vae_args = NestedDictToClass(curve_vae_cfg)
    wireset_vae_args = NestedDictToClass(wireset_vae_cfg)

    curve_model_type = vae_cfg.curve_model_type
    wireframe_model_type = vae_cfg.wireframe_model_type

    is_curve_recon = vae_cfg.is_curve_recon == 1
    main(
        curve_vae_args, 
        wireset_vae_args, 
        device_idx=0, 
        is_curve_recon=is_curve_recon,
        curve_model_type=curve_model_type,
        wireframe_model_type=wireframe_model_type,
    )
