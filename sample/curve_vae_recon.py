import torch
import os
import sys
import numpy as np
from einops import rearrange, repeat
from pathlib import Path
import logging
from torch.utils.data import DataLoader

from model_loader import load_curve_vae
from utils import denorm_curves, polylines_to_png

from src.utils.config import NestedDictToClass
from src.dataset.dataset_fn import curve_to_mean_custom_collate
from src.dataset.dataset import WireframeNormDataset

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

def save_vtx_adjs_zs(
    vertices: np.ndarray,
    adjs: np.ndarray,
    zs: np.ndarray,
    uuid: str,
    tgt_dir_path: str,
    save_only_zs: bool = False,
):
    group_id = uuid[:3]
    chunck_id = uuid[:4]
    uid = uuid.split('_')[0]
    tgt_dir_path = Path(tgt_dir_path).joinpath(group_id, chunck_id, uid)
    tgt_dir_path.mkdir(parents=True, exist_ok=True)
    
    if save_only_zs:
        tgt_file_path = tgt_dir_path.joinpath(f'{uuid}.npy')
        np.save(tgt_file_path, zs)
    else:
        tgt_file_path = tgt_dir_path.joinpath(f'{uuid}.npz')
        np.savez(
            tgt_file_path, 
            vertices=vertices,
            adjs=adjs,
            zs=zs,
        )


def curve_recon(
    curve_vae_cfg: NestedDictToClass, 
    curve_model_type: str = 'ae',
    device: str = 'cuda:0',
    logger: logging.Logger = None
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
        batch_size=128,
        shuffle=False,
        num_workers=32 if curve_vae_cfg.batch_size > 10 else curve_vae_cfg.batch_size, # use smaller of 10 or batch_size as num_workers
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
                
                vertices_i = vertices_list[i]
                adjs_i = adjs_list[i]
                
                zs = np.concatenate([mu_i, std_i], axis=-1)
                
                save_vtx_adjs_zs(
                    vertices_i, adjs_i, zs, 
                    uid, curve_vae_cfg.data.recon_dir_path, save_only_zs=False)
        else:
            raise ValueError(f"Invalid curve model type: {curve_model_type}")


        logger.info(f"batch {batch_idx} done.")

    exit()