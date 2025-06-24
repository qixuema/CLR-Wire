import os
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
from einops import rearrange

from src.flow.utils import parse_ode_args, parse_sde_args, parse_transport_args
from src.transport import create_transport
from src.transport.transport import Sampler
from src.utils.config import NestedDictToClass, load_config
from src.utils.helpers import setup_logging, generate_random_string
from sample.model_loader import load_flow, load_wireframe_vae, load_curve_vae
from sample.reconstruction import recon_and_save_wireframe_from_logits


logger = logging.getLogger(__name__)
setup_logging()

device_idx = 0
device = f'cuda:{device_idx}' if torch.cuda.is_available() else "cpu"

# Setup PyTorch:
torch.manual_seed(1234)
torch.set_grad_enabled(False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(flow_args, wireframe_vae_args, curve_vae_args, mode_args):

    tgt_dir_path = flow_args.data.gen_path + '/' + datetime.now().strftime("%Y_%m%d_%H%M_%S")

    os.makedirs(tgt_dir_path, exist_ok=True)

    logger.info("load Flow Matching model...")
    flow_model = load_flow(flow_args, device)
    wireframe_model = load_wireframe_vae(wireframe_vae_args, model_type='de', device=device)
    curve_model = load_curve_vae(curve_vae_args, model_type='de', device=device)
    
    transport = create_transport(
        mode_args.path_type,
        mode_args.prediction,
        mode_args.loss_weight,
        mode_args.train_eps,
        mode_args.sample_eps
    )
    
    sampler = Sampler(transport)
    if flow_args.sampling.mode == "ODE":
        if mode_args.likelihood:
            assert mode_args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=mode_args.sampling_method,
                num_steps=mode_args.num_sampling_steps,
                atol=mode_args.atol,
                rtol=mode_args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=mode_args.sampling_method,
                num_steps=mode_args.num_sampling_steps,
                atol=mode_args.atol,
                rtol=mode_args.rtol,
                reverse=mode_args.reverse
            )
            
    elif flow_args.sampling.mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=mode_args.sampling_method,
            diffusion_form=mode_args.diffusion_form,
            diffusion_norm=mode_args.diffusion_norm,
            last_step=mode_args.last_step,
            last_step_size=mode_args.last_step_size,
            num_steps=mode_args.num_sampling_steps,
        )

    bs = flow_args.sampling.batch_size

    for i in range(flow_args.sampling.iter_num):
        # Create sampling noise:
        noise = torch.randn(
            bs, 
            flow_args.model.latent_num, 
            flow_args.model.in_channels,
            device=device
        )

        context = None
        model_kwargs = dict(context=context)
        
        with torch.no_grad():
            zs = sample_fn(
                noise,
                flow_model.forward,
                **model_kwargs,
            )[-1]
        
        with torch.no_grad():
            preds = wireframe_model(
                zs = zs,
                sample_posterior=True,
                return_loss=False, 
                return_std=False,
            )
        
        pred_curve_latent = preds['curve_latent']

        assert pred_curve_latent.shape[-1] == 12

        pred_curve_latent = rearrange(pred_curve_latent, "b n (d c) -> (b n) d c", d = 4)
        pred_curve_latent = rearrange(pred_curve_latent, "b d c -> b c d")

        with torch.no_grad():                
            dec_curves = curve_model(z=pred_curve_latent).sample
        
        dec_curves = dec_curves.detach().cpu().numpy()

        # Fix boundary points
        dec_curves[:, :, 0] = np.array([-1, 0, 0])
        dec_curves[:, :, -1] = np.array([1, 0, 0])
        
        # rearrange
        dec_curves = rearrange(dec_curves, "(b n) c d -> b n c d", b=bs) # batch size, num curves, num points, 3
        dec_curves = rearrange(dec_curves, "b n c d -> b n d c") 

        filenames = generate_random_string(15, bs)
        res = recon_and_save_wireframe_from_logits(
            recon_curve=True,
            pred_logits=preds,
            filenames=filenames,
            tgt_dir_path=tgt_dir_path,
            dec_curves=dec_curves,
            check_valid=True,
            logger=logger,
            threshold=1e-3,
        )

        logger.info(f"batch {i} completed.")

    exit()    

def parse_args():
    

    # Arguments
    parser = argparse.ArgumentParser(description='Train a line autoencoder model.')
    parser.add_argument('--curve_vae_config', type=str, help='Path to curve vae config file.',
                        default="src/configs/train_curve_vae.yaml")
    parser.add_argument('--wireframe_vae_config', type=str, help='Path to wireframe vae config file.',
                        default="src/configs/train_wireframe_vae.yaml")
    parser.add_argument('--flow_config', type=str, help='Path to flow config file.',
                        default="src/configs/train_flow_matching.yaml")
    
    args, unknown = parser.parse_known_args()
    
    return args


if __name__ == "__main__":
    
    
    cfgs = parse_args()
    curve_vae_cfg = load_config(cfgs.curve_vae_config)
    wireset_vae_cfg = load_config(cfgs.wireframe_vae_config)
    flow_cfg = load_config(cfgs.flow_config)

    curve_vae_args = NestedDictToClass(curve_vae_cfg)
    wireset_vae_args = NestedDictToClass(wireset_vae_cfg)
    flow_args = NestedDictToClass(flow_cfg)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    
    parse_transport_args(parser)
    
    mode = flow_args.sampling.mode
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    mode_args = parser.parse_known_args()[0]

    main(flow_args, wireset_vae_args, curve_vae_args, mode_args)
