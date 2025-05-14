from math import sqrt
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from x_transformers.x_transformers import AttentionLayers
from diffusers.models.unets.unet_1d_blocks import ResConvBlock, SelfAttention1d, Upsample1d
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import BaseOutput

import einx
from einops import reduce, pack, rearrange
from einops.layers.torch import Reduce, Rearrange

from src.utils.helpers import exists, default, divisible_by

class Embedder(Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)

class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,
        min_dim = 16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min = 1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = PixelNorm(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out, dropout = dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask)
        h = self.excite(h, mask = mask)
        return h + res

@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"

# ==================== MLP Mixer ====================

def MLP(in_dim, out_dim = None, expansion_factor = 4.0, dropout = 0.0, dense = nn.Linear):
    if out_dim is None:
        out_dim = in_dim
        
    inner_dim = int(in_dim * expansion_factor)
   
    return nn.Sequential(
        dense(in_dim, inner_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        dense(inner_dim, out_dim),
    )

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, input_channels, out_channels, dim, depth, num_lines=256, expansion_factor=4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_lines, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Linear(dim, out_channels)
    )

# ==================== Permutator ====================

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))

def Permutator(*, image_size, patch_size, dim, depth, num_classes, segments, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b w c (h s)', s = s),
                        nn.Linear(height * s, height * s),
                        Rearrange('b w c (h s) -> b h w (c s)', s = s),
                    ),
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b h c (w s)', s = s),
                        nn.Linear(width * s, width * s),
                        Rearrange('b h c (w s) -> b h w (c s)', s = s),
                    ),
                    nn.Linear(dim, dim)
                ),
                nn.Linear(dim, dim)
            )),
            PreNormResidual(dim, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * expansion_factor, dim),
                nn.Dropout(dropout)
            ))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b h w c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class AttentionLayerFactory:
    @staticmethod
    def create_transformer_encoder_layer(dim: int, heads: int, depth: int, dropout: float = 0.0):
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout=dropout)
        return TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth)

    @staticmethod
    def create_transformer_decoder_layer(dim: int, heads: int, depth: int, dropout: float = 0.0):
        decoder_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout= dropout)
        return TransformerDecoder(decoder_layer=decoder_layer, num_layers=depth)

    @staticmethod
    def create_x_transformer_cross_attn_layer(dim: int, heads: int, depth: int):
        return AttentionLayers(
            dim=dim, heads=heads, depth=depth, 
            cross_attend=True, only_cross=True,
            attn_flash=True,
        )
        
    @staticmethod
    def create_x_transformer_self_attn_layer(dim: int, heads: int, depth: int):
        return AttentionLayers(
            dim=dim, heads=heads, depth=depth, 
            attn_flash=True,
        )

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, other_dim=0):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        # self.mlp = nn.Linear(self.embedding_dim+3, dim)/
        self.mlp = MPConv(self.embedding_dim+3+other_dim, dim, kernel=[])

    @staticmethod
    def embed(input, basis):
        # print(input.shape, basis.shape)
        projections = torch.einsum('nd,de->ne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=1)
        return embeddings
    
    def forward(self, input):
        # input: N x 3
        if input.shape[1] != 3:
            input, others = input[:, :3], input[:, 3:]
        else:
            others = None
        
        if others is None:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=1)) # N x C
        else:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input, others], dim=1))
        return embed



# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(
        self,
        times, # (b, n)
    ): 
        # output shape is (b, n, self.dim + 1)

        freqs = einx.multiply('... i, j -> ... i j', times, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((times, freqs.sin(), freqs.cos()), 'b n *')
        return fourier_embed



class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, temb=None): 
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, task_type='multi-class', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets, num_classes=None, label_smoothing=0.0, class_weights=None, reduction='none'):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        
        return self.multi_class_focal_loss(inputs, targets, num_classes, label_smoothing, class_weights, reduction)


    def multi_class_focal_loss(self, inputs, targets, num_classes=None, label_smoothing=0.0, class_weights=None, reduction='none'):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            
        if num_classes is None:
            num_classes = self.num_classes

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        if targets_one_hot.ndim == 3:
            targets_one_hot = rearrange(targets_one_hot, 'b n c -> b c n')

        # Compute cross-entropy for each class
        # ce_loss = -targets_one_hot * torch.log(probs)
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        
        ce_loss = F.cross_entropy(
            inputs, 
            targets,
            reduction='none', 
            label_smoothing=label_smoothing, 
            weight=class_weights
        )
        
        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight * ce_loss

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


def compute_first_and_second_col(col_diff_logits, row_diff_logits):
    # Convert logits to probabilities
    col_diff_probs = F.softmax(col_diff_logits, dim=-1)
    row_diff_probs = F.softmax(row_diff_logits, dim=-1)
    
    # Compute first_col by cumulative sum of col_diff along the last axis (axis=-1)
    first_col_logits = torch.cumsum(col_diff_probs, dim=-1)
    
    # Compute second_col by adding row_diff_probs to first_col_logits
    second_col_logits = first_col_logits + row_diff_probs
    
    return first_col_logits, second_col_logits