import os
import argparse
import torch
import sys
import logging

# Add the parent directory of this file to Python’s module search path,
# allowing imports from modules located in the project’s root directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from src.utils.config import load_config, NestedDictToClass
from src.utils.helpers import setup_logging
from curve_vae_recon import curve_recon


logger = logging.getLogger(__name__)
setup_logging()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            logger=logger,
        )
    else:
        wireset_recon(
            wireframe_vae_cfg, 
            curve_vae_cfg, 
            wireframe_model_type=wireframe_model_type,
            device=device, 
            logger=logger,
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
