from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear
from typing import Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.unets.unet_1d_blocks import get_down_block
from x_transformers.x_transformers import AttentionLayers

from einops import rearrange

from src.vae.modules import AutoencoderKLOutput, RandomFourierEmbed, UNetMidBlock1D, UpBlock1D
from src.utils.torch_tools import interpolate_1d, calculate_polyline_lengths, sample_edge_points, point_seq_tangent

class Encoder1D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock1D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        sample_points_num=32,
        act_fn="silu",
        double_z=True,
        use_tangent=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv1d(
            in_channels = in_channels + 3 if use_tangent else in_channels,
            out_channels = block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.cross_attend = AttentionLayers(
            dim = block_out_channels[0],
            depth=2,
            heads = 8,
            cross_attend=True,
            use_rmsnorm=True,
            attn_flash=True,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )
        
        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv1d(block_out_channels[-1], conv_out_channels, 3, padding=1)
        
        self.gradient_checkpointing = False

        self.sample_points_num = sample_points_num

    def forward(self, x): 
        tangents = point_seq_tangent(x, channel_dim=-2, seq_dim=-1)
        x = torch.cat([x, tangents], dim=-2)

        x = self.conv_in(x)

        # cross for global information
        B, C, num_points = x.shape
        indices = torch.linspace(0, num_points - 1, self.sample_points_num, dtype=int)
        sample = x[:, :, indices]
        # from conv style to seq style
        x = rearrange(x, 'b c n -> b n c')
        sample = rearrange(sample, 'b c n -> b n c')
        sample = self.cross_attend(sample, context=x)
        # back to conv style
        sample = rearrange(sample, 'b n c -> b c n')

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)[0]
        
        # middle
        sample = self.mid_block(sample)
       
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample
    

class Decoder1D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock1D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            
            up_block = UpBlock1D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        
        self.conv_act = nn.SiLU()

        self.query_embed = nn.Sequential(
            RandomFourierEmbed(block_out_channels[0]),
            Linear(block_out_channels[0] + 1, block_out_channels[0]),
            nn.SiLU()
        )

        self.cross_attend = AttentionLayers(
            dim = block_out_channels[0],
            depth=2,
            heads = 8,
            cross_attend=True,
            attn_flash=True,
            only_cross=True,
        )

        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False


    def forward(self, z, queries, latent_embeds=None):
        sample = self.conv_in(z)

        # middle
        sample = self.mid_block(sample, latent_embeds)
        # sample = sample.to(upscale_dtype)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        
        # cross-attention
        sample = rearrange(sample, 'b d n -> b n d')

        queries_embeddings = self.query_embed(queries)
        sample = self.cross_attend(queries_embeddings, context=sample)

        sample = rearrange(sample, 'b n d -> b d n')

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        return sample

class AutoencoderKL1D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock1D",),
        up_block_types: Tuple[str] = ("UpBlock1D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 64,
        kl_weight: float = 1e-6,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            sample_points_num=sample_points_num,
        )

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv =  nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv =  nn.Conv1d(latent_channels, latent_channels, 1)

        self.sample_points_num = sample_points_num

        self.kl_weight = kl_weight

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
       
        moments = self.quant_conv(h) 
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        z = self.post_quant_conv(z)
        dec = self.decoder(z, t)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        decoded = self._decode(z, t).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        data: torch.FloatTensor,
        t: Optional[torch.FloatTensor] = None,
        sample_posterior: bool = False,  # True
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        data = rearrange(data, "b n c -> b c n") # for conv1d input

        bs = data.shape[0]

        # indices = torch.arange(0, 64, 2)
        # x = data[:, :, indices]

        posterior = self.encode(data).latent_dist
        
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        
        if t is None:
            t = torch.rand(bs, self.sample_points_num, device=data.device)
            t, _ = torch.sort(t, dim=-1)
        else:
            assert t.shape[1] == self.sample_points_num, "t should have the same number of self.sample_points_num"

        dec = self.decode(z, t).sample

        if not return_dict:
            return (dec,)
        
        if return_loss:
            kl_loss = 0.5 * torch.sum(
                torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar,
                dim=[1, 2],
            ).mean()

            gt_samples = interpolate_1d(t, data)

            data = rearrange(data, 'b c n -> b n c')
            batch_lengths = calculate_polyline_lengths(data)
            batch_lengths = torch.clamp(batch_lengths, min=2.0, max=torch.pi * 10)

            weights = torch.log(batch_lengths + 0.2) # through log to reduce the influence of long polylines
            batch_loss = F.mse_loss(dec, gt_samples, reduction='none').mean(dim=[1, 2])
            recon_loss = (batch_loss * weights).mean()
            loss = recon_loss + self.kl_weight * kl_loss

            return loss, dict(
                recon_loss = recon_loss,
                kl_loss = kl_loss
            )

        return DecoderOutput(sample=dec)

class AutoencoderKL1DFastEncode(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock1D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv =  nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1)

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
       
        moments = self.quant_conv(h) 
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def forward(
        self,
        data: torch.FloatTensor,
        return_std: bool = False,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        data = sample_edge_points(data, 32) # should we downsample the data to 32 points?

        data = rearrange(data, "b n c -> b c n") # for conv1d input

        posterior = self.encode(data).latent_dist
        z = posterior.mode()

        if return_std:
            return z, posterior.std

        return z

class AutoencoderKL1DFastDecode(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        out_channels: int = 3,
        up_block_types: Tuple[str] = ("UpBlock1D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.post_quant_conv =  nn.Conv1d(latent_channels, latent_channels, 1)
  
    def _decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, t)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor = None,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        if t is None:
            device = z.device
            bs = z.shape[0]
            t = torch.linspace(0, 1, 32, device=device).repeat(bs, 1)

        decoded = self._decode(z, t)

        return decoded # this is a class, you need to use .sample to get the sample
   
    