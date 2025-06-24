import torch
from typing import Any
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

from src.flow.flow import DiT


def load_wireframe_vae(args: Any, model_type: str, device: torch.device):
    """
    Load the WireSet VAE model or WireSet VAE Decoder.
    """
    
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
        wireframe_latent_num=args.model.wireframe_latent_num,
        label_smoothing=args.label_smoothing,
        cls_loss_weight=args.model.cls_loss_weight,
        segment_loss_weight=args.model.segment_loss_weight,
        col_diff_loss_weight=args.model.col_diff_loss_weight,
        row_diff_loss_weight=args.model.row_diff_loss_weight,
        curve_latent_loss_weight=args.model.curve_latent_loss_weight,
        kl_loss_weight=args.model.kl_loss_weight,
        curve_latent_embed_dim=args.model.curve_latent_embed_dim,
        use_mlp_predict=args.model.use_mlp_predict,
        use_latent_pos_emb=args.model.use_latent_pos_emb,
    )
    checkpoint_path = f"{args.model.checkpoint_folder}/{args.model.checkpoint_file_name}"
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt, strict=strict)
    model = model.to(device)
    model.eval()
    return model


def load_curve_vae(args: Any, model_type: str, device: torch.device):
    """
    Load the Curve VAE model or Curve VAE Decoder.
    """
    if 'ae' in model_type:
        Model = AutoencoderKL1D
        strict = True
    elif 'de' in model_type:
        Model = AutoencoderKL1DFastDecode
        strict = False
    elif 'en' in model_type:
        Model = AutoencoderKL1DFastEncode
        strict = False
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
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt, strict=strict)
    model = model.to(device)
    model.eval()
    return model


def load_flow(args: Any, device: torch.device):
    """
    Load the Flow Matching model.
    """

    model = DiT(
        in_channels=args.model.in_channels,
        latent_num=args.model.latent_num,
        hidden_size=args.model.hidden_size,
        num_heads=args.model.num_heads,
        depth=args.model.depth,
        mlp_ratio=args.model.mlp_ratio,
        condition_on_points=args.model.condition_on_points,
        condition_on_img=args.model.condition_on_img,
    )
    
    checkpoint_path = f"{args.model.checkpoint_folder}/{args.model.checkpoint_file_name}"
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()    
    
    return model
    