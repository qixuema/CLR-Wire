import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import math
from timm.models.vision_transformer import Mlp, Attention

from src.vae.vae_wireframe import AutoencoderKLWireframeFastEncode

try:
    from src.pointnet2.pointnet2.models.pointnet2_msg_cls import PointNet2ClassificationMSG_WOPL
except:
    print("PointNet2ClassificationMSG_WOPL not found")


def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and condition                    #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, output_size=None, frequency_embedding_size=256):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Block                                    #
#################################################################################

class Block(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=16,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        learn_sigma=False,
        latent_num = 128,
        condition_on_points = False,
        use_pointnet = True,
        condition_on_img = False,
        use_dinov2 = False,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.latent_num = latent_num
        self.cond_feature_dim = hidden_size
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        

        self.t_embedder = TimestepEmbedder(hidden_size, output_size=hidden_size)
      
        self.pos_embed = nn.Parameter(torch.randn(latent_num, hidden_size))

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        
        self.condition_on_img = condition_on_img
        self.condition_on_points = condition_on_points
        
        if condition_on_img:
            self.use_dinov2 = use_dinov2
            if use_dinov2:
                img_latent_channels = 1024
                self.process_img_latent = nn.Sequential(
                    Mlp(
                        in_features=img_latent_channels, 
                        hidden_features=1024,
                        out_features=self.cond_feature_dim,
                        act_layer=nn.Sigmoid, 
                        norm_layer=nn.LayerNorm,
                        drop=0.5
                    ),
                    nn.Sigmoid()
                )
            else:
                raise ValueError("Invalid condition_on_img mode")
            
        elif condition_on_points:
            self.use_pointnet = use_pointnet
            
            if use_pointnet:
                args = {
                    "model.use_xyz": True,
                }
                self.process_shape_latent = PointNet2ClassificationMSG_WOPL(args)
            else:
                raise ValueError("Invalid condition_on_points mode")

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)    

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, context=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """


        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        if self.condition_on_points:
            c = self.process_shape_latent(context)
            if self.use_pointnet:
                c = t + c
        elif self.condition_on_img:
            c = self.process_img_latent(context)
            if self.use_dinov2:
                c = t + c
            else:
                raise ValueError("Invalid condition_on_img mode")
        else:
            c = t

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
                    
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)

        return x

    def forward_with_cfg(self, x, t, context, cfg_scale):

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[..., :self.in_channels], model_out[..., self.in_channels:]
        eps = model_out
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=-1)
        return half_eps
