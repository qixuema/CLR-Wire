import torch
import json
from torch import is_tensor
import os
import numpy as np
from einops import rearrange
import logging
from torch.utils.data import DataLoader

from model_loader import load_wireset_vae, load_curve_vae
from src.utils.config import NestedDictToClass
from src.dataset.dataset import WireframeDataset, LatentDataset
from reconstruction import recon_and_save_wireframe_from_logits

def wireframe_recon(
    wireframe_vae_cfg: NestedDictToClass, 
    curve_vae_cfg: NestedDictToClass, 
    wireframe_model_type: str = 'en', 
    device: str = 'cuda:0',
    logger: logging.Logger = None
):

    # load models
    logger.info("load Wireframe VAE model...")
    wireframe_model = load_wireset_vae(wireframe_vae_cfg, wireframe_model_type, device)
    
    # ==================================================================================
    
    if wireframe_model_type != 'en':
        logger.info("load Curve VAE model...")
        curve_model_type = 'de' # or 'decoder' or 'ae' or 'encoder'
        curve_model = load_curve_vae(curve_vae_cfg, curve_model_type, device)

    # Ensure target directory exists
    os.makedirs(wireframe_vae_cfg.data.recon_dir_path, exist_ok=True)

    # load dataset
    logger.info("load dataset...")
    if wireframe_model_type == 'de':
        dataset = LatentDataset(
            dataset_file_path=wireframe_vae_cfg.data.test_set_file_path,
            condition_on_points=False,
            sample=True,
        )
    else:
        dataset = WireframeDataset(
            dataset_file_path=wireframe_vae_cfg.data.test_set_file_path,
            sample=True,
            max_num_lines=wireframe_vae_cfg.model.max_curves_num,
        )


    logger.info("Initialize dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=wireframe_vae_cfg.batch_size,
        num_workers=24 if wireframe_vae_cfg.batch_size > 10 else wireframe_vae_cfg.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    
    dl_iter = dataloader

    if wireframe_model_type == 'en':
        zs = []
    
    if wireframe_model_type == 'en':
        recon_curve = False
    else:
        recon_curve = True

    if recon_curve:
        encode_wireframe_latent = False
    else:
        encode_wireframe_latent = True
    

    logger.info("start reconstruction...")
    logger.info(f"num_iter: {len(dataloader)}")
    
    res_all = {}
    for batch_idx, data in enumerate(dl_iter):
        uids = data['uid']
        batch_size = len(uids)
        
        forward_kwargs = {k: v.to(device) for k, v in data.items() if is_tensor(v)} 

        with torch.no_grad():
            preds = wireframe_model(
                **forward_kwargs,
                # sample_posterior=True,
                return_loss=False, 
                return_std=True,
            )

        if wireframe_model_type == 'en':
            zs.append(preds)
        
            if encode_wireframe_latent:
                preds = preds.detach().cpu().numpy()
                # save each sample's latent
                for j, uid in enumerate(uids):
                    tgt_file_path = os.path.join(wireframe_vae_cfg.data.recon_dir_path, f"{uid}.npz")
                    if os.path.exists(tgt_file_path):
                        continue
                    
                    wireset_latent = dict(
                        uid = uid,
                        zs = rearrange(preds[j], "c n -> n c")
                    )
                    np.savez(tgt_file_path, **wireset_latent)

                continue
            
            continue

        dec_curves = None
        if recon_curve:     
            
            pred_curve_latent = preds['curve_latent']

            # TEST: use gt to cover the predicted curves
            # pred_curve_latent = forward_kwargs['xs'][..., 6:18]

            assert pred_curve_latent.shape[-1] == 12

            pred_curve_latent = rearrange(pred_curve_latent, "b n (d c) -> (b n) d c", d = 4)
            pred_curve_latent = rearrange(pred_curve_latent, "b d c -> b c d")

            # Perform reconstruction
            with torch.no_grad():                
                dec_curves = curve_model(z=pred_curve_latent).sample
            
            dec_curves = dec_curves.detach().cpu().numpy()

            # Fix boundary points
            dec_curves[:, :, 0] = np.array([-1, 0, 0])
            dec_curves[:, :, -1] = np.array([1, 0, 0])
            
            # rearrange
            dec_curves = rearrange(dec_curves, "(b n) c d -> b n c d", b=batch_size) # batch size, num curves, num points, 3
            dec_curves = rearrange(dec_curves, "b n c d -> b n d c") 

        res = recon_and_save_wireframe_from_logits(
            pred_logits=preds,
            uids=uids,
            tgt_dir_path=wireframe_vae_cfg.data.recon_dir_path,
            recon_curve=recon_curve,
            dec_curves=dec_curves,
            forward_kwargs=forward_kwargs,
            # use_adj_refine_segment_logits=False,
            # use_gt_diff_and_segment=True,
            check_valid=True, #TODO:
            # save_adj=True,
            logger=logger,
        )

        res_all.update(res)
        
        logger.info(f"batch {batch_idx} completed.")
    
        exit()
    
    # save res_all as a json file
    with open(os.path.join(wireframe_vae_cfg.data.recon_dir_path, "res_all.json"), "w") as f:
        json.dump(res_all, f)

    
    if encode_wireframe_latent:
        exit()

    if wireframe_model_type == 'en':
        mus = torch.cat(mus, dim=0)
        mus = mus.detach().cpu().numpy()
        np.save(os.path.join(wireframe_vae_cfg.data.recon_dir_path, "mus.npy"), mus)

    logger.info("Reconstruction completed.")

