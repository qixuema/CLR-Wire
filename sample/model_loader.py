# models/model_loader.py

import torch
from typing import Any
import sys
print(sys.path)
from src.utils.misc import load_ema
from src.vae.vae_wireframe import (
    AutoencoderKLWireframe, 
    AutoencoderKLWireframeFastEncode, 
    AutoencoderKLWireframeFastDecode
)

from src.vae.vae_curve import (
    AutoencoderKL1D,
    AutoencoderKL1DFastEncode, 
    AutoencoderKL1DFastDecode,
)

from typing import Optional, Union
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

def load_wireset_vae(args: Any, model_type: str, device: torch.device):
    """
    Load the WireSet VAE model or WireSet VAE Decoder.
    """
    train_continuous_wireset = args.train_continuous_wireset if hasattr(args, 'train_continuous_wireset') else False
    
    if 'ae' in model_type:
        Model = AutoencoderKLWireframe
        strict = True
    elif 'de' in model_type:
        Model = AutoencoderKLWireframeFastDecode
        strict = False
    elif 'en' in model_type:
        Model = AutoencoderKLWireframeFastEncode
        strict = False
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    model = Model(
        latent_channels=args.model.latent_channels,
        attn_encoder_depth=args.model.attn_encoder_depth,
        attn_decoder_self_depth=args.model.attn_decoder_self_depth,
        attn_decoder_cross_depth=args.model.attn_decoder_cross_depth,
        attn_dim=args.model.attn_dim,
        num_heads=args.model.num_heads,
        max_row_diff=args.model.max_row_diff,
        max_col_diff=args.model.max_col_diff,
        max_curves_num=args.model.max_curves_num,
        wireset_latent_num=args.model.wireset_latent_num,
        label_smoothing=args.label_smoothing,
        flag_bce_loss_weight=args.model.flag_bce_loss_weight,
        segment_ce_loss_weight=args.model.segment_ce_loss_weight,
        col_diff_ce_loss_weight=args.model.col_diff_ce_loss_weight,
        row_diff_ce_loss_weight=args.model.row_diff_ce_loss_weight,
        curve_latent_loss_weight=args.model.curve_latent_loss_weight,
        kl_loss_weight=args.model.kl_loss_weight,
        use_pos_embedding=args.model.use_pos_embedding,
        curve_latent_embed_dim=args.model.curve_latent_embed_dim,
        use_mlp_predict=args.model.use_mlp_predict,
        fix_std=args.model.fix_std if hasattr(args.model, 'fix_std') else False,
        use_flag_embed=args.model.use_flag_embed if hasattr(args.model, 'use_flag_embed') else False,
        is_64=args.model.is_64 if hasattr(args.model, 'is_64') else False,
    )
    checkpoint_path = f"{args.model.checkpoint_folder}/{args.model.checkpoint_file_name}"
    model = load_ema(model, checkpoint_path, strict=strict)
    model = model.to(device)
    model.eval()
    return model


def load_curve_vae(args: Any, model_type: str, device: torch.device):
    """
    Load the Curve VAE model or Curve VAE Decoder.
    """
    if 'ae' in model_type:
        Model = AutoencoderKL1D
    elif 'de' in model_type:
        Model = AutoencoderKL1DFastDecode
    elif 'en' in model_type:
        Model = AutoencoderKL1DFastEncode
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    
    model = Model(
        in_channels=args.model.in_channels,
        out_channels=args.model.out_channels,
        down_block_types=args.model.down_block_types,
        up_block_types=args.model.up_block_types,
        block_out_channels=args.model.block_out_channels,
        layers_per_block=args.model.layers_per_block,
        act_fn=args.model.act_fn,
        latent_channels=args.model.latent_channels,
        norm_num_groups=args.model.norm_num_groups,
        sample_points_num=args.model.sample_points_num,
    )
    checkpoint_path = f"{args.model.checkpoint_folder}/{args.model.checkpoint_file_name}"
    model = load_ema(model, checkpoint_path, strict=False)
    model = model.to(device)
    model.eval()
    return model
