from typing import Dict, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from x_transformers.x_transformers import AttentionLayers
from beartype import beartype
from beartype.typing import Tuple, Optional

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from torchtyping import TensorType
from einops import rearrange, repeat, pack

from src.vae.modules import AutoencoderKLOutput, MLP
from src.utils.helpers import (
    exists, 
    is_odd, 
    default, 
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

class EmbeddingLayerFactory:
    def __init__(
        self, 
        num_discrete_coors: int, 
        point_embed_dim: int, 
        max_col_diff: int, 
        col_diff_embed_dim: int,
        max_row_diff: int, 
        row_diff_embed_dim: int, 
    ):
        self.num_discrete_coors = num_discrete_coors
        self.point_embed_dim = point_embed_dim
        self.max_col_diff = max_col_diff
        self.col_diff_embed_dim = col_diff_embed_dim
        self.max_row_diff = max_row_diff
        self.row_diff_embed_dim = row_diff_embed_dim

    def create_embeddings(self):
        
        point_embed = PointEmbed(dim=self.point_embed_dim)
        
        col_diff_embed = nn.Embedding(self.max_col_diff, self.col_diff_embed_dim)
        row_diff_embed = nn.Embedding(self.max_row_diff, self.row_diff_embed_dim)
        return point_embed, col_diff_embed, row_diff_embed


# attention_layer_factory.py
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
            use_rmsnorm=True, 
            attn_flash=True,
        )
        
    @staticmethod
    def create_x_transformer_self_attn_layer(dim: int, heads: int, depth: int):
        return AttentionLayers(
            dim=dim, heads=heads, depth=depth, 
            use_rmsnorm=True, 
            attn_flash=True,
        )


class Encoder1D(Module):
    def __init__(
        self,
        out_channels = 8,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        coor_embed_dim = 128,
        max_col_diff=6,
        max_row_diff=32,
        col_diff_embed_dim = 16,
        row_diff_embed_dim = 32,
        max_vertices_num = 192,
        max_curves_num = 128,
        curve_latent_channels = 12,
        curve_latent_embed_dim = 256,
        attn_kwargs: dict = dict(
            dim = 512,
            depth = 4,
            heads = 8,
        ),        
        double_z = True,
        wireframe_latent_num = 64,
    ):
        super().__init__()
      
        self.max_curves_num = max_curves_num
        self.wireframe_latent_num = wireframe_latent_num

        embedding_factory = EmbeddingLayerFactory(
            num_discrete_coors = num_discrete_coors,
            point_embed_dim = coor_embed_dim * 3,
            max_col_diff = max_col_diff,
            col_diff_embed_dim = col_diff_embed_dim,
            max_row_diff = max_row_diff,
            row_diff_embed_dim = row_diff_embed_dim,
        )

        (
            self.point_embed, 
            self.col_diff_embed, 
            self.row_diff_embed
        ) = embedding_factory.create_embeddings()

        attn_dim = attn_kwargs['dim']
        
        # latent mu embedding
        
        self.latent_embed = nn.Linear(curve_latent_channels, curve_latent_embed_dim)

        self.enc_learnable_queries = nn.Parameter(torch.randn(wireframe_latent_num, attn_dim))            
        
        # position embedding
        
        self.pos_emb = nn.Parameter(torch.randn(max_curves_num, attn_dim))

        init_dim = (coor_embed_dim * 6 + col_diff_embed_dim + row_diff_embed_dim + curve_latent_embed_dim)

        self.attn_project_in = nn.Linear(init_dim, attn_dim)

        attn_factory = AttentionLayerFactory()
        self.cross_attn = attn_factory.create_x_transformer_cross_attn_layer(**attn_kwargs)
                    
        out_channels = 2 * out_channels if double_z else out_channels

        self.project_out = nn.Linear(attn_dim, out_channels)
        
        # register buffer
        self.register_buffer('max_col_diff', torch.tensor(max_col_diff))
        self.register_buffer('max_row_diff', torch.tensor(max_row_diff))
        self.register_buffer('max_vertices_num', torch.tensor(max_vertices_num))

    def forward(
        self,
        *,
        xs:                     TensorType['b', 'nv', 6+12, float],
        flag_diffs:             TensorType['b', 'nl', 1+2, int],  # as edges
        return_segment_coords:  bool = False
    ):
        bs = xs.shape[0]
        
        line_coords = xs[...,:6] # bs, nl, 6
        
        points = rearrange(line_coords, 'b nl (nlv d) -> b nl nlv d', nlv = 2)
        points = rearrange(points, 'b nl nlv d -> (b nl nlv) d')

        line_coor_embed = self.point_embed(points) #  (bs, nl, 6, dim_coor_embed)
        line_coor_embed = rearrange(line_coor_embed, '(b nl nlv) d -> b nl nlv d', b=bs, nlv=2)
        line_coor_embed = rearrange(line_coor_embed, 'b nl nlv d -> b nl (nlv d)')
                
        flag = flag_diffs[..., 0].unsqueeze(-1) # bs, nl, 1
        diffs = flag_diffs[..., 1:] # bs, nl, 2
        
        col_diff = diffs[..., 0]
        row_diff = diffs[..., 1]
    
        col_diff_embed = self.col_diff_embed(col_diff)
        row_diff_embed = self.row_diff_embed(row_diff)

        curve_latent = xs[..., 6:] # bs, nv, 12
        curve_latent_embed = self.latent_embed(curve_latent) # bs, nv, 128
        
        wire_embed, _ = pack([line_coor_embed, col_diff_embed, row_diff_embed, curve_latent_embed], 'b nl *')

        wire_embed = self.attn_project_in(wire_embed)
        
        wire_embed = self.pos_emb + wire_embed
        
        # multi cross Attention

        # ==================== use learnable ====================
        
        enc_learnable_query = repeat(self.enc_learnable_queries, 'n d -> b n d', b=wire_embed.shape[0])

        # ==================== cross attention ====================
        
        context_padding_mask = rearrange(flag < 0.5, 'b n c -> b (n c)')

        wireframe_latent_embed = self.cross_attn(x=enc_learnable_query, context=wire_embed, context_mask=~context_padding_mask)
        
        # project out

        wireframe_latent = self.project_out(wireframe_latent_embed)

        if not return_segment_coords:
            return wireframe_latent
        
        return wireframe_latent

class Decoder1D(Module):
    def __init__(
        self, 
        *,
        in_channels: int = 8,
        attn_kwargs: dict = dict(
            dim = 512,
            heads = 8,
            self_depth = 6,
            cross_depth = 2,
        ),
        max_curves_num = 256,
    ):
        super().__init__()
        
        self.max_curves_num = max_curves_num

        attn_dim = attn_kwargs['dim']

        self.proj_in = nn.Linear(in_channels, attn_dim)

        self.dec_learnable_query = nn.Parameter(torch.randn(max_curves_num, attn_dim))            
        
        attn_factory = AttentionLayerFactory()
        
        self.self_attn = attn_factory.create_x_transformer_self_attn_layer(
            dim=attn_kwargs['dim'],
            heads=attn_kwargs['heads'], 
            depth=attn_kwargs['self_depth']
        )
        
        self.cross_attn = attn_factory.create_x_transformer_cross_attn_layer(
            dim=attn_kwargs['dim'],
            heads=attn_kwargs['heads'], 
            depth=attn_kwargs['cross_depth']
        )
        
        self.proj_out = nn.Linear(attn_dim, attn_dim)

    @beartype
    def forward(
        self,
        zs: TensorType['b', 'n', 'd', float],
    ):
        bs = zs.shape[0]
        wireframe_latent = self.proj_in(zs)

        # self attn
        
        # x = rearrange(x, 'b n d -> n b d')
        wireframe_latent = self.self_attn(wireframe_latent)
        
        # cross attn
        
        query_embed = repeat(self.dec_learnable_query, 'n d -> b n d', b=bs)
        
        query_embed = self.cross_attn(query_embed, wireframe_latent)

        # proj out
        
        query_embed = self.proj_out(query_embed)

        return query_embed

class AutoencoderKLWireframe(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 8,
        max_col_diff=6,
        max_row_diff=32,
        attn_encoder_depth: int = 4,
        attn_decoder_self_depth: int = 6,
        attn_decoder_cross_depth: int = 2,
        attn_dim: int = 512,
        num_heads: int = 8,
        max_curves_num: int = 128,
        wireframe_latent_num: int = 64,
        label_smoothing: float = 0.005,
        flag_bce_loss_weight: float = 1.,
        segment_ce_loss_weight: float = 1.,
        col_diff_ce_loss_weight: float = 1.,
        row_diff_ce_loss_weight: float = 1.,
        curve_latent_loss_weight: float = 1.,
        kl_loss_weight: float = 2e-4,
        curve_latent_embed_dim: int = 256,
        use_mlp_predict: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.max_col_diff = max_col_diff
        self.max_row_diff = max_row_diff
        self.max_curves_num = max_curves_num

        attn_kwargs = dict(
            dim = attn_dim,
            heads = num_heads,
            depth = attn_encoder_depth,
        )

        self.encoder = Encoder1D(
            out_channels=latent_channels,
            attn_kwargs=attn_kwargs,
            max_curves_num=max_curves_num,
            wireframe_latent_num=wireframe_latent_num,
            curve_latent_embed_dim=curve_latent_embed_dim,
        )

        attn_kwargs = dict(
            dim = attn_dim,
            heads = num_heads,
            self_depth = attn_decoder_self_depth,
            cross_depth = attn_decoder_cross_depth,
        )

        self.decoder = Decoder1D(
            in_channels=latent_channels, 
            attn_kwargs=attn_kwargs,
            max_curves_num=max_curves_num,
        )
        
        
        self.use_mlp_predict = use_mlp_predict
        if use_mlp_predict:
            dim = attn_dim

            self.predict_flag = MLP(in_dim=dim, out_dim=1)
            self.predict_diffs = MLP(in_dim=dim, out_dim=6+32)
            self.predict_segments = MLP(in_dim=dim, out_dim=768)
            self.predict_curve_latent = MLP(in_dim=dim, out_dim=12)        
                        
        else:
            out_dim = 1 + 6 + 6 + 32 + 12 # num_flags + num_segments + num_col_diffs + num_row_diffs + num_curve_latent
            self.linear_predict_logits = nn.Linear(attn_dim, out_dim)
        

        self.quant_proj = nn.Linear(2 * latent_channels, 2 * latent_channels)
        self.post_quant_proj = nn.Linear(latent_channels, latent_channels)

        # for loss function
        self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        self.flag_bce_loss_weight = flag_bce_loss_weight
        self.segment_mse_loss_weight = segment_ce_loss_weight
        self.col_diff_ce_loss_weight = col_diff_ce_loss_weight
        self.row_diff_ce_loss_weight = row_diff_ce_loss_weight
        self.curve_latent_loss_weight = curve_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight

        self.pad_id = -1
        
        self.label_smoothing = label_smoothing

        col_diff_labels = torch.linspace(-1, 1, self.max_col_diff)
        col_diff_class_weights = torch.exp(col_diff_labels)
        
        row_diff_labels = torch.linspace(-1, 1, self.max_row_diff)
        row_diff_class_weights = torch.exp(row_diff_labels)

        # 设置权重        
        t = torch.linspace(0, 2, self.max_curves_num)
        col_weights = 1.2 - 0.2 * torch.log(t + 1.7183)

        start_segment = 128*6 + 1
        start_col = start_segment + self.max_col_diff
        start_curve_latent = start_col + 12

        self.register_buffer('col_diff_class_weights', col_diff_class_weights)
        self.register_buffer('row_diff_class_weights', row_diff_class_weights)
        self.register_buffer('col_weights', col_weights)
        self.register_buffer('start_segment', torch.tensor(start_segment))
        self.register_buffer('start_col', torch.tensor(start_col))
        self.register_buffer('start_curve_latent', torch.tensor(start_curve_latent))

    @apply_forward_hook
    def encode(
        self, 
        *,
        xs:                 TensorType['b', 'nl', 6, float],
        flag_diffs:              TensorType['b', 'nl', 2, int],
        return_dict:        bool = True,
        return_segment_coords: bool = True,
    ) -> AutoencoderKLOutput:

        h = self.encoder(
            xs=xs, 
            flag_diffs=flag_diffs, 
        )
        # assert not torch.isnan(h).any(), "h is NaN"

        moments = self.quant_proj(h) 

        moments = rearrange(moments, 'b n d -> b d n')

        # assert not torch.isnan(moments).any(), "moments is NaN"

        posterior = DiagonalGaussianDistribution(moments)
        
        if not return_dict:
            return (posterior,)


        if not return_segment_coords:
            return AutoencoderKLOutput(latent_dist=posterior)
        
        return AutoencoderKLOutput(
            latent_dist=posterior, 
        )

    def _decode(
        self, 
        *,
        zs:              TensorType['b', 'nl', 'd', float],
        return_dict:    bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        zs = rearrange(zs, 'b d n -> b n d')
        zs = self.post_quant_proj(zs)
        dec = self.decoder(zs)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # @apply_forward_hook
    def decode(
        self, 
        *,
        z:              TensorType['b', 'nl', 'd', float], 
        return_dict:    bool = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        decoded = self._decode(zs=z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)


    def linear_predict(self, dec):
        num_flags = 1
        num_segments = num_flags + 6
        num_diffs = num_segments + 6 + 32

        pred_logits = self.linear_predict_logits(dec)

        pred_flag_logits = pred_logits[..., :num_flags]
        pred_segments = pred_logits[..., num_flags:num_segments]
        assert pred_segments.shape[-1] == 6
        pred_diffs_logits = pred_logits[..., num_segments:num_diffs]
        assert pred_diffs_logits.shape[-1] == 38
        pred_curve_latent = pred_logits[..., num_diffs:]
        assert pred_curve_latent.shape[-1] == 12
        
        return pred_flag_logits, pred_segments, pred_diffs_logits, pred_curve_latent

    def mlp_predict(self, dec):
        # for multi task
        pred_flag_logits = self.predict_flag(dec)
        pred_segment_logits = self.predict_segments(dec)
        pred_diffs_logits = self.predict_diffs(dec)
        pred_curve_latent = self.predict_curve_latent(dec)
        
        return pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent

    def loss(
        self, 
        *,
        segment_coords,
        flag_diffs,
        gt_curve_latent,
        xs_mask,
        preds,
    ):
        (
            pred_flag_logits, 
            pred_segments, 
            pred_diffs_logits, 
            pred_curve_latent_mu,
        ) = (
            preds['flags'],
            preds['segments'],
            preds['diffs'],
            preds['curve_latent'],
        )
        
        bs = pred_flag_logits.shape[0]

        flag = flag_diffs[..., 0]
        diffs = flag_diffs[..., 1:]

        # ================== flag bce loss =================
        pred_flag_logits = pred_flag_logits.squeeze(-1)
        flag_bce_loss = F.binary_cross_entropy_with_logits(
            pred_flag_logits, 
            flag.float(), 
            reduction='mean'
        )

        # ================== segment ce loss =================


        segment_mse_loss = self.mse_loss_fn(pred_segments, segment_coords)

        line_mask = repeat(xs_mask, 'b nl -> b nl r', r = 6)
        segment_mse_loss = segment_mse_loss[line_mask].mean()

        # ================== col diff and row diff ce loss =================
        
        rearranged_logits = rearrange(pred_diffs_logits, 'b ... c -> b c (...)')
        
        pred_col_diff_logits, pred_row_diff_logits = rearranged_logits.split([self.max_col_diff, rearranged_logits.shape[1] - self.max_col_diff], dim=1)

        col_diff_ce_loss = F.cross_entropy(
            pred_col_diff_logits, 
            diffs[..., 0],
            reduction='none', 
            label_smoothing=self.label_smoothing, 
            weight=self.col_diff_class_weights
        )
        
        row_diff_ce_loss = F.cross_entropy(
            pred_row_diff_logits, 
            diffs[..., 1],
            reduction='none', 
            label_smoothing=self.label_smoothing, 
            weight=self.row_diff_class_weights
        )
                    
        col_weights = repeat(self.col_weights, 'n -> b n', b=bs)
                    
        col_diff_ce_loss = (col_diff_ce_loss * col_weights)[xs_mask].mean()
        row_diff_ce_loss = row_diff_ce_loss[xs_mask].mean()

        # ================== curve latent mse loss =================
        gt_curve_latent_std = torch.clamp(gt_curve_latent[..., 12:], 0., 1.)
        mu_weights = 1.2 - 0.5 * torch.log(gt_curve_latent_std + 1.7183)
        
        curve_latent_mask = repeat(xs_mask, 'b nl -> b nl r', r = 12)
        curve_latent_loss = (mu_weights * self.mse_loss_fn(pred_curve_latent_mu, gt_curve_latent[..., :12]))[curve_latent_mask].mean()        

        return flag_bce_loss, segment_mse_loss, col_diff_ce_loss, row_diff_ce_loss, curve_latent_loss, None

    def forward(
        self,
        xs: TensorType['b', 'nl', 6+12+12, float],
        flag_diffs: TensorType['b', 'nl', 1+2, int],
        sample_posterior: bool = False,  # True
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            xs: segment coordinates and curve latent
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """

        # prepare masks
        flags = flag_diffs[..., 0]
        xs_mask = flags > 0.5

        # encode and get posterior
        ae_kl_output = self.encode(
            xs=xs[..., :18], # 6 + 12
            flag_diffs=flag_diffs, 
        )

        posterior = ae_kl_output.latent_dist
        segment_coords = xs[..., :6]
        
        # sample_posterior = False
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # decode
        dec = self.decode(z=z).sample

        if self.use_mlp_predict:
            pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent = self.mlp_predict(dec)
        else:
            pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent = self.linear_predict(dec)

        preds = {
            'flags': pred_flag_logits,
            'segments': pred_segment_logits,
            'diffs': pred_diffs_logits,
            'curve_latent': pred_curve_latent,
        }


        if not return_dict:
            return (preds,)

        if return_loss:
            kl_loss = 0.5 * torch.sum(torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar, dim = [1,2]).mean()
            
            if not sample_posterior:
                kl_loss = 0 * kl_loss
            
            (
                flag_bce_loss, 
                segment_mse_loss, 
                col_diff_ce_loss, 
                row_diff_ce_loss,
                curve_latent_loss,
                L2_loss,
            ) = self.loss(
                segment_coords=segment_coords,
                flag_diffs=flag_diffs,
                gt_curve_latent = xs[..., 6:],
                xs_mask=xs_mask,
                preds=preds,
            )

            a, b, c = (0.25, 0.733, 6.955)
            
            new_kl_loss_weight = self.kl_loss_weight
            
            loss = (self.flag_bce_loss_weight*flag_bce_loss
                + self.segment_mse_loss_weight*segment_mse_loss
                + self.col_diff_ce_loss_weight*col_diff_ce_loss 
                + self.row_diff_ce_loss_weight*row_diff_ce_loss 
                + self.curve_latent_loss_weight*curve_latent_loss
                + new_kl_loss_weight*kl_loss
            )

            return loss, dict(
                flag_bce_loss=flag_bce_loss,
                segment_mse_loss=segment_mse_loss,
                col_diff_ce_loss=col_diff_ce_loss,
                row_diff_ce_loss=row_diff_ce_loss,
                curve_latent_loss=curve_latent_loss,
                kl_loss=kl_loss,
            )
        
        return preds


class AutoencoderKLWireframeFastEncode(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 8,
        max_col_diff=6,
        max_row_diff=32,
        attn_encoder_depth: int = 4,
        attn_dim: int = 512,
        num_heads: int = 8,
        max_curves_num: int = 128,
        wireframe_latent_num: int = 64,
        curve_latent_embed_dim: int = 256,
        use_mlp_predict: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.max_col_diff = max_col_diff
        self.max_row_diff = max_row_diff
        self.max_curves_num = max_curves_num

        attn_kwargs = dict(
            dim = attn_dim,
            heads = num_heads,
            depth = attn_encoder_depth,
        )

        self.encoder = Encoder1D(
            out_channels=latent_channels,
            attn_kwargs=attn_kwargs,
            max_curves_num=max_curves_num,
            wireframe_latent_num=wireframe_latent_num,
            curve_latent_embed_dim=curve_latent_embed_dim,
        )

        self.quant_proj = nn.Linear(2 * latent_channels, 2 * latent_channels)

    @apply_forward_hook
    def encode(
        self, 
        *,
        xs:                 TensorType['b', 'nl', 6, float],
        flag_diffs:              TensorType['b', 'nl', 2, int],
        return_dict:        bool = True,
        return_segment_coords: bool = True,
    ) -> AutoencoderKLOutput:

        h = self.encoder(
            xs=xs, 
            flag_diffs=flag_diffs, 
        )

        moments = self.quant_proj(h) 

        moments = rearrange(moments, 'b n d -> b d n')

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)


        if not return_segment_coords:
            return AutoencoderKLOutput(latent_dist=posterior)
        
        return AutoencoderKLOutput(
            latent_dist=posterior, 
        )

    def forward(
        self,
        xs: TensorType['b', 'nl', 6, float],
        flag_diffs: TensorType['b', 'nl', 2, int],
        sample_posterior: bool = False,  # True
        generator: Optional[torch.Generator] = None,
        return_std: bool = False,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            xs: segment coordinates and curve latent
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """

        # encode and get posterior
        ae_kl_output = self.encode(
            xs=xs[..., :18], 
            flag_diffs=flag_diffs, 
        )

        posterior = ae_kl_output.latent_dist
        
        # sample_posterior = False
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        
        if return_std:
            std = posterior.std
            zs = torch.cat([z, std], dim=1)
            return zs
        return z


class AutoencoderKLWireframeFastDecode(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 8,
        max_col_diff=6,
        max_row_diff=32,
        attn_decoder_self_depth: int = 6,
        attn_decoder_cross_depth: int = 2,
        attn_dim: int = 512,
        num_heads: int = 8,
        max_curves_num: int = 128,
        wireframe_latent_num: int = 64,
        use_mlp_predict: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.max_col_diff = max_col_diff
        self.max_row_diff = max_row_diff
        self.max_curves_num = max_curves_num

        attn_kwargs = dict(
            dim = attn_dim,
            heads = num_heads,
            self_depth = attn_decoder_self_depth,
            cross_depth = attn_decoder_cross_depth,
        )

        self.decoder = Decoder1D(
            in_channels=latent_channels, 
            curveset_latent_num=wireframe_latent_num,
            attn_kwargs=attn_kwargs,
            max_curves_num=max_curves_num,
        )

        self.use_mlp_predict = use_mlp_predict
        if use_mlp_predict:
            dim = attn_dim

            self.predict_flag = MLP(in_dim=dim, out_dim=1)
            self.predict_diffs = MLP(in_dim=dim, out_dim=6+32)
            self.predict_segments = MLP(in_dim=dim, out_dim=768)
            self.predict_curve_latent = MLP(in_dim=dim, out_dim=12)        
                        
        else:
            out_dim = 1 + 6 + 6 + 32 + 12 # num_flags + num_segments + num_col_diffs + num_row_diffs + num_curve_latent
            self.linear_predict_logits = nn.Linear(attn_dim, out_dim)

        self.post_quant_proj = nn.Linear(latent_channels, latent_channels)

        start_segment = 128*6 + 1
        start_col = start_segment + self.max_col_diff
        start_curve_latent = start_col + 12

        self.register_buffer('start_segment', torch.tensor(start_segment))
        self.register_buffer('start_col', torch.tensor(start_col))
        self.register_buffer('start_curve_latent', torch.tensor(start_curve_latent))

    def _decode(
        self, 
        *,
        zs:              TensorType['b', 'nl', 'd', float],
        return_dict:    bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        zs = rearrange(zs, 'b d n -> b n d')
        zs = self.post_quant_proj(zs)
        dec = self.decoder(zs)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, 
        *,
        zs:              TensorType['b', 'nl', 'd', float], 
        return_dict:    bool = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        decoded = self._decode(zs=zs).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def linear_predict(self, dec):
        num_flags = 1
        num_segments = num_flags + 6
        num_diffs = num_segments + 6 + 32

        pred_logits = self.linear_predict_logits(dec)

        pred_flag_logits = pred_logits[..., :num_flags]
        pred_segments = pred_logits[..., num_flags:num_segments]
        assert pred_segments.shape[-1] == 6
        pred_diffs_logits = pred_logits[..., num_segments:num_diffs]
        assert pred_diffs_logits.shape[-1] == 38
        pred_curve_latent = pred_logits[..., num_diffs:]
        assert pred_curve_latent.shape[-1] == 12
        
        return pred_flag_logits, pred_segments, pred_diffs_logits, pred_curve_latent

    def mlp_predict(self, dec):
        # for multi task
        pred_flag_logits = self.predict_flag(dec)
        pred_segment_logits = self.predict_segments(dec)
        pred_diffs_logits = self.predict_diffs(dec)
        pred_curve_latent = self.predict_curve_latent(dec)
        
        return pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent


    def forward(
        self,
        zs: TensorType['b', 'nwl', 16, float],
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            xs: segment coordinates and curve latent
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        zs = rearrange(zs, 'b nwl d -> b d nwl')

        # decode
        dec = self.decode(zs=zs).sample

        if self.use_mlp_predict:
            pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent = self.mlp_predict(dec)
        else:
            pred_flag_logits, pred_segment_logits, pred_diffs_logits, pred_curve_latent = self.linear_predict(dec)

        preds = {
            'flags': pred_flag_logits,
            'segments': pred_segment_logits,
            'diffs': pred_diffs_logits,
            'curve_latent': pred_curve_latent,
        }
        
        if not return_dict:
            return (dec,)

        return preds
