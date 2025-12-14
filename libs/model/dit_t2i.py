# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

from turtle import st
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import open_clip
import torch.utils.checkpoint

from .trans_autoencoder import TransEncoder, Adaptor

from ipdb import set_trace as st


def modulate(x, shift, scale):
    if shift.ndim < x.ndim:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
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
    CrossFlow: update it for CFG with indicator
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        embeddings = self.embedding_table(labels.int())
        return embeddings


class LabelEmbedder2(nn.Module):
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
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_cross_attn=False, disable_cross_attn=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.use_cross_attn = use_cross_attn
        self.disable_cross_attn = disable_cross_attn
        if self.use_cross_attn and not self.disable_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=0.0,
                batch_first=True,
                kdim=hidden_size,
                vdim=hidden_size,
            )
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(hidden_size, hidden_size * 6, bias=True),
            nn.Linear(hidden_size, hidden_size * (9 if self.use_cross_attn and not self.disable_cross_attn else 6), bias=True),
        )

    def forward(self, x, c, cond_x=None, cond_mask=None):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, cond_x, cond_mask)
        # return self._forward(x, c, cond_x)

    def _forward(self, x, c, cond_x=None, cond_mask=None):
        # st() 
        if self.use_cross_attn and not self.disable_cross_attn:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_ca, scale_ca, gate_ca = self.adaLN_modulation(c).chunk(9, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1) # Note -1 
        if gate_msa.ndim < 3:
            gate_msa = gate_msa.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
            if self.use_cross_attn and not self.disable_cross_attn:
                gate_ca = gate_ca.unsqueeze(1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_cross_attn and not self.disable_cross_attn:
            x = x + gate_ca * self.cross_attn(modulate(self.norm_ca(x), shift_ca, scale_ca), cond_x, cond_x, key_padding_mask=cond_mask)[0] 
            # x = x + self.cross_attn(self.norm_ca(x), cond_x, cond_x, key_padding_mask=cond_mask)[0] 
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # st()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) # Note -1 
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=2, # for cfg indicator
    ):
        super().__init__()
        self.input_size = config.latent_size
        self.learn_sigma = config.learn_sigma
        self.in_channels = config.channels
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.edit_mode = config.edit_mode if hasattr(config, "edit_mode") else False
        self.cond_mode = config.cond_mode if hasattr(config, "cond_mode") else None
        assert self.cond_mode in [None, 'channel', 'cross-attn', 'self-attn']
        if self.edit_mode and self.cond_mode == 'channel':
            self.in_channels *= 2
        self.direct_map = config.direct_map if hasattr(config, "direct_map") else False
        self.use_cross_attn = config.use_cross_attn if hasattr(config, "use_cross_attn") else False
        self.do_regular_cfg = config.do_regular_cfg if hasattr(config, "do_regular_cfg") else False 
        self.disable_cross_attn = config.disable_cross_attn if hasattr(config, "disable_cross_attn") else False

        self.x_embedder = PatchEmbed(self.input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if not self.do_regular_cfg:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.do_class_cond = hasattr(config, "do_class_cond") and config.do_class_cond
        if self.do_class_cond:
            assert self.disable_cross_attn
            self.c_embedder = LabelEmbedder(num_classes, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=self.use_cross_attn, disable_cross_attn=self.disable_cross_attn) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.use_textVE = not (self.edit_mode and self.cond_mode == 'cross-attn' and self.direct_map and self.use_cross_attn)
        if self.use_textVE:

            ######### CrossFlow related
            if hasattr(config.textVAE, "num_down_sample_block"):
                down_sample_block = config.textVAE.num_down_sample_block
            else:
                down_sample_block = 3
            self.context_encoder = TransEncoder(d_model=config.clip_dim, N=config.textVAE.num_blocks, num_token=config.num_clip_token,
                                                head_num=config.textVAE.num_attention_heads, d_ff=config.textVAE.hidden_dim, 
                                                latten_size=config.channels * config.latent_size * config.latent_size * 2, 
                                                down_sample_block=down_sample_block, dropout=config.textVAE.dropout_prob, last_norm=False)


            self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=None) # Why not use pretrained???? 
            self.open_clip_output = Adaptor(input_dim=1024, 
                                        tar_dim=config.channels * config.latent_size * config.latent_size
                                        )
            del self.open_clip.text
            del self.open_clip.logit_bias

        else:
            if not self.disable_cross_attn : 
                self.cond_embedder = nn.Sequential(
                    # nn.LayerNorm(config.clip_dim),
                    nn.Linear(config.clip_dim, hidden_size, bias=True),
                )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if not self.do_regular_cfg:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        if self.do_class_cond:
            nn.init.normal_(self.c_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _forward(self, x, t, null_indicator, cond_image=None, cond_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        # if self.edit_mode and self.direct_map:
        #     # st()
        #     x, cond_image = cond_image, x # 
        # BUG BUG BUG !!!!!!!! Should not only switch here -- earlier schedule, intermediate target etc. need to be calculated based on the switched variables !!!!!! BUG BUG BUG TODO TODO TODO 

        if self.edit_mode and self.cond_mode == 'channel':
            # st()
            assert cond_image is not None
            x = torch.cat([x, cond_image], dim=1)  # (N, 2*C, H, W)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # Altho also masked but has been masked in TextVE so always unified shape (&same as img) here? 
        t = self.t_embedder(t)                   # (N, D)
        if not self.do_regular_cfg:
            y = self.y_embedder(null_indicator)    # (N, D)
        # Note that this is only an indicator for cfg, not the real cond, as the real con is x already in CF\ 

        if self.do_class_cond:
            y = self.c_embedder(cond_image)      # cond_image should be text prompt and then should be class labels 

        if self.edit_mode and self.cond_mode == 'self-attn':
            # st()
            cond_x = self.x_embedder(cond_image) + self.pos_embed # 
            x = torch.cat([x, cond_x], dim=1)

        if self.edit_mode and self.cond_mode == 'cross-attn' and self.use_textVE:
            assert cond_image is not None
            cond_x = self.x_embedder(cond_image) + self.pos_embed  # (N, T, D)
            if not self.use_cross_attn :
                # st()
                y = y.unsqueeze(1)  # (N, 1, D)
                # y = torch.cat([y, cond_x], dim=-2)  # (N, 1+T, D)
                y = y + cond_x
                # No dedicated Cross Attn (sequential concat) here, so have to do AdaLN (turn to spatial), or have to implement additional Cross Attn (sequential concat) from scratch! 
                # (This might also be one of advantages of CF, that no sequential concat Cross Attn is needed! (not bcuz of compact save by dit, but bcuz of no cond any more by direct mapping!) -- (and btw not highlighted, so my temp attn etc. also)
                t = t.unsqueeze(1)  # (N, 1, D)

        if (not self.do_regular_cfg) or self.do_class_cond : 
            c = t + y                                # (N, D)
        else:
            c = t

        if not self.use_textVE and not self.disable_cross_attn : 
            # st()
            cond_x = self.cond_embedder(cond_image)  # (N, L, D)
            assert cond_mask is not None 
            cond_mask = cond_mask.bool()
        else:
            cond_x = cond_mask = None  

        for block in self.blocks:
            if self.edit_mode and self.cond_mode == 'cross-attn' and self.use_cross_attn : 
                x = block(x, c, cond_x=cond_x, cond_mask=cond_mask )                      # (N, T, D)
            else:
                x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        if self.edit_mode and self.cond_mode == 'self-attn':
            # st()
            x, cond_x = x.chunk(2, dim=1)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return [x]

    def _forward_with_cfg(self, x, t, cfg_scale, cond_image=None, cond_mask=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, None, cond_image=cond_image, cond_mask=cond_mask)[0]
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _text_encoder(self, condition_context, tar_shape, mask):
        output = self.context_encoder(condition_context, mask)
        mu, log_var = torch.chunk(output, 2, dim=-1)        
        z = self._reparameterize(mu, log_var)

        return [z, mu, log_var]

    def _img_clip(self, image_input):
        image_latent = self.open_clip.encode_image(image_input)
        image_latent = self.open_clip_output(image_latent)

        return image_latent, self.open_clip.logit_scale

    def forward(self, x, t = None, log_snr = None, text_encoder=False, text_decoder=False, image_clip=False, shape=None, mask=None, null_indicator=None, cond_image=None, cond_mask=None, cfg_scale=0.0 ):
        if text_encoder:
            return self._text_encoder(condition_context = x, tar_shape=shape, mask=mask)
        elif text_decoder:
            raise NotImplementedError
            return self._text_decoder(condition_enbedding = x, tar_shape=shape) # mask is not needed for decoder
        elif image_clip:
            return self._img_clip(image_input = x) 
        elif cfg_scale > 0.0:
            return self._forward_with_cfg(x, t, cfg_scale, cond_image=cond_image, cond_mask=cond_mask)
        else:
            return self._forward(x = x, t = t, null_indicator=null_indicator, cond_image=cond_image, cond_mask=cond_mask)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_H_2(config, **kwargs):
    return DiT(config=config, depth=36, hidden_size=1280, patch_size=2, num_heads=20, **kwargs)

def DiT_XL_2(config, **kwargs):
    return DiT(config=config, depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(config, **kwargs):
    return DiT(config=config, depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(config, **kwargs):
    return DiT(config=config, depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(config, **kwargs):
    return DiT(config=config, depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(config, **kwargs):
    return DiT(config=config, depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(config, **kwargs):
    return DiT(config=config, depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(config, **kwargs):
    return DiT(config=config, depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}