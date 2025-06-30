import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from ..model_madiff.drop import DropPath
import sys
import numpy as np
from functools import partial
# diffusion
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import UniformSampler, MaskedDiffusionSampler, SNRSampler
from guided_diffusion.script_util import create_gaussian_diffusion

from ..model_madiff.util import *

# ver12
# no cls token, pool instead
# follow SODA
# feature modulation: AdaGN
# layer mask
# source add gaussian
# inverse cosine scheduler

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        x = self.forward_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale

        #attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1,2).reshape(B, N, C*self.num_heads)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # assert 'stage' in st_mode
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# AdaGN(AdaLN)
# currently use LayerNorm, not GroupNorm
class FeatureModulation(nn.Module):
    def __init__(self, dim, dim_t_embed, layer_mask_ratio, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.dim_t_embed = dim_t_embed
        self.fc_z = nn.Linear(dim, dim * 2)
        self.fc_t = nn.Linear(dim_t_embed, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.layer_mask_ratio = layer_mask_ratio

    def forward(self, x, *, z, t):
        '''
        x: [N, L, C]
        z: [N, L, C]
        t: [N, C']
        '''
        if z is None or np.random.rand(1)[0] < self.layer_mask_ratio:
            z = torch.zeros_like(x)

        z = self.fc_z(z)
        zs, zb = z[:,:,:self.dim], z[:,:,self.dim:]
        t = self.fc_t(t)
        ts, tb = t[:,None,:self.dim], t[:,None,self.dim:]
        x = zs * (ts * self.norm(x) + tb) + zb

        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, dim_t_embed=64, # useless
                 mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 layer_mask_ratio=0.,
                 ):
        super().__init__()

        self.layer_mask_ratio = layer_mask_ratio

        self.norm1 = FeatureModulation(dim, dim_t_embed, layer_mask_ratio) ###
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = FeatureModulation(dim, dim_t_embed, layer_mask_ratio) ###

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, *, t, z, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x, z=z, t=t), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x, z=z, t=t)))

        return x
    
class SkeleEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        dim_in=3,
        dim_feat=256,
        num_frames=120,
        num_joints=25,
        patch_size=1,
        t_patch_size=4,
    ):
        super().__init__()
        assert num_frames % t_patch_size == 0
        num_patches = (
            (num_joints // patch_size) * (num_frames // t_patch_size)
        )
        self.input_size = (
            num_frames // t_patch_size,
            num_joints // patch_size
        )
        print(
            f"num_joints {num_joints} patch_size {patch_size} num_frames {num_frames} t_patch_size {t_patch_size}"
        )

        self.num_joints = num_joints
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = num_joints // patch_size
        self.t_grid_size = num_frames // t_patch_size

        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        _, T, V, _ = x.shape
        x = torch.einsum("ntsc->ncts", x)  # [N, C, T, V]
        
        assert (
            V == self.num_joints
        ), f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert (
            T == self.num_frames
        ), f"Input skeleton length ({T}) doesn't match model ({self.num_frames})."
        
        x = self.proj(x)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, V, C]
        return x

class Transformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, dim_t_embed=64,
                 layer_mask_ratio=0.,
                 uncond_ratio=0.,
                 depth=5, decoder_depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 act_layer=nn.GELU, ###
                 drop_path_rate=0., norm_layer=nn.LayerNorm, #norm_skes_loss=False,
                 diff_prediction='noise', diff_steps=1000, diff_noise_schedule='cosine',
                 input_mean=[0.,0.,0.], 
                 input_var=[1.,1.,1.],
                 lambda_loss_uni=0.,
                 one_person=True,
                 loss_reweight=None, # ['MinSNR', 5]
                 self_shift=False,
                 ):

        super().__init__()
        print('#'*50)
        print('Activate Motion-aware masking if motion_aware_tau > 0.')

        self.lambda_loss_uni = lambda_loss_uni
        print(f'Uniformity loss lambda: {lambda_loss_uni}')
        self.one_person = one_person

        self.dim_feat = dim_feat
        self.dim_t_embed = dim_t_embed
        self.layer_mask_ratio = layer_mask_ratio

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.uncond_ratio = uncond_ratio

        print(f'Layer mask ratio: {layer_mask_ratio}, Uncond ratio: {uncond_ratio}')
        
        if isinstance(act_layer, str): act_layer = eval(act_layer)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, 
                act_layer=act_layer,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                layer_mask_ratio=layer_mask_ratio, 
                act_layer=act_layer,
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(dim_feat)

        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.decoder_temp_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

        self.decoder_pred = nn.Linear(
            dim_feat,
            t_patch_size * patch_size * dim_in,
            bias=True
        ) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Initialize weights
        self.apply(self._init_weights)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # DDPM specifics
        self.diff_prediction = diff_prediction
        self.diff_noise_schedule = diff_noise_schedule
        self.diff_steps = diff_steps
        self.input_mean = input_mean
        self.input_var = input_var
        self.input_std = [np.sqrt(var) for var in input_var]
        self.self_shift = self_shift
        if self_shift: self.input_mean = [0,0,0]

        print(f'Use self-shift: {self_shift}, input_mean: {self.input_mean}, input_var: {self.input_var}, input_std: {self.input_std}')
        
        self.diffusion = create_gaussian_diffusion(
            learn_sigma=False,
            steps=self.diff_steps,
            noise_schedule=self.diff_noise_schedule,
            timestep_respacing="",
            use_kl=False,
            predict_xstart=(self.diff_prediction == 'joint'),
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        self.schedule_sampler = UniformSampler(self.diffusion)
        #self.schedule_sampler = MaskedDiffusionSampler(self.diffusion, 0, 750)
        #self.schedule_sampler = SNRSampler(self.diffusion, gamma=0.3)
        
        if loss_reweight is None:
            self.loss_reweight = torch.ones(self.diff_steps)
        elif loss_reweight[0] == 'MinSNR':
            self.loss_reweight = get_MinSNR_weights(self.diffusion, gamma=loss_reweight[1])
        elif loss_reweight[0] == 'MaxSNR':
            self.loss_reweight = get_MaxSNR_weights(self.diffusion, gamma=loss_reweight[1])
        else:
            raise NotImplementedError

        print('Loss reweight:', {i:self.loss_reweight[i] for i in range(self.diff_steps)})     
        print('#'*50)     
        # --------------------------------------------------------------------------  
        #self.m1, self.m2, self.m3 = AvgMeter(), AvgMeter(), AvgMeter()
        #self.v1, self.v2, self.v3 = AvgMeter(), AvgMeter(), AvgMeter()

    def _init_weights(self, m, gain=1.):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                print('Warning! LayerNorm has no weights!')
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = round(L * (1 - mask_ratio))
        
        #if np.random.rand(1)[0] < 0.001: print(f'Random masking, total len {L}, len_keep {len_keep}')

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_keep

    def motion_aware_random_masking(self, x, *, x_orig, mask_ratio, tau):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [NM, L, D], sequence
        x_orig: patchified original skeleton sequence
        """
        NM, L, D = x.shape  # batch, length, dim
        _, TP, VP, _ = x_orig.shape
        
        len_keep = round(L * (1 - mask_ratio))

        x_orig_motion = torch.zeros_like(x_orig)
        x_orig_motion[:, 1:, :, :] = torch.abs(x_orig[:, 1:, :, :] - x_orig[:, :-1, :, :])
        x_orig_motion[:, 0, :, :] = x_orig_motion[:, 1, :, :]
        x_orig_motion = x_orig_motion.mean(dim=[3])  # NM, TP, VP
        x_orig_motion = x_orig_motion.reshape(NM, L)

        x_orig_motion = x_orig_motion / (torch.max(x_orig_motion, dim=-1, keepdim=True).values * tau + 1e-10)
        x_orig_motion_prob = F.softmax(x_orig_motion, dim=-1) 

        # anti-motion aware
        noise = torch.log(x_orig_motion_prob) - torch.log(-torch.log(torch.rand(NM, L, device=x.device) + 1e-10) + 1e-10)  # gumble

        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_remove = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([NM, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_keep

    def forward_encoder(self, x, *, x_orig, mask_ratio, motion_aware_tau):
        # embed skeletons
        x = self.joints_embed(x)

        NM, TP, VP, _ = x.shape

        # add pos & temp embed
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]
         
        # masking: length -> length * mask_ratio
        x = x.reshape(NM, TP * VP, -1)
        
        if motion_aware_tau <= 0.:
            x, mask, ids_restore, _ = self.random_masking(x, mask_ratio)
        else:
            x_orig = self.patchify(x_orig).reshape(shape=(NM, TP, VP, -1))
            x, mask, ids_restore, _ = self.motion_aware_random_masking(x, x_orig=x_orig, mask_ratio=mask_ratio, tau=motion_aware_tau)

        # apply Transformer blocks
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        latent = self.norm(x)

        cls_token = latent.mean(dim=1, keepdim=True)

        return latent, cls_token, mask, ids_restore

    def forward_decoder(self, x, *, latent, cls_token, t, ids_restore):
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size
        C = self.dim_feat
        L = latent.shape[1]

        # embed tokens
        z = torch.cat([latent, cls_token.repeat(1,TP*VP-L,1)], dim=1)  # no cls token
        z = torch.gather(
            z, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )  # unshuffle

        uncond_mask = (torch.rand(NM, device=x.device) > self.uncond_ratio).float() # 1 keep 0 drop
        z = z * uncond_mask[:,None,None]

        t = timestep_embedding(t, self.dim_t_embed)

        x = self.decoder_embed(x)

        # add pos & temp embed
        x = x + self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]  # NM, TP, VP, C
        
        # apply Transformer blocks
        x = x.reshape(NM, TP * VP, C)

        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x, z=z, t=t)
        
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)

        return x

    def patchify(self, imgs):
        """
        imgs: (N, T, V, 3)
        x: (N, L, t_patch_size * patch_size * 3)
        """
        NM, T, V, C = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert V % p == 0 and T % u == 0
        VP = V // p
        TP = T // u

        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))
        x = torch.einsum("ntuvpc->ntvupc", x)
        x = x.reshape(shape=(NM, TP * VP, u * p * C))
        return x
    
    def forward_loss(self, imgs, pred, mask, t):
        """
        imgs: [NM, T, V, 3]
        pred: [NM, TP * VP, t_patch_size * patch_size * 3]
        mask: [NM, TP * VP], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # [NM, TP * VP, C]

        loss = (pred - target) ** 2

        #print(loss.mean(dim=0).mean(dim=-1).reshape(30,25).sum(dim=0))

        if np.random.rand(1)[0] < 0.002:
            print('pred',(mask[:,:,None]*pred)[0])
            print('target',(mask[:,:,None]*target)[0])
            print(f't = {t[0]}, loss = {(loss.mean(dim=-1) * mask)[0].sum() / mask[0].sum()}')
        
        loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch

        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)
        weights = self.loss_reweight[t].to(pred.device)
        loss = (loss * weights).mean()

        return loss.mean()

    def forward(self, x, x_aug, mask_ratio=0.80, motion_stride=1, motion_aware_tau=0.75, **kwargs):

        with torch.no_grad():
            #only use 1 person
            if self.one_person:
                x, x_aug = x[...,0:1], x_aug[...,0:1]
                #x, x_aug = remove_zero_data(x, x_aug)

            N, C, T, V, M = x.shape # NTU

            x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
            x_orig = x.clone().detach()
            
            x_aug = x_aug.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)

            if self.self_shift: 
                mu = x.mean(dim=[1,2], keepdim=True)
                x = x - mu
                x_aug = x_aug - mu

            for i in range(C):
                x[:,:,:,i] = (x[:,:,:,i] - self.input_mean[i]) / self.input_std[i]
                x_aug[:,:,:,i] = (x_aug[:,:,:,i] - self.input_mean[i]) / self.input_std[i]
           
            x_gt = x.clone().detach()

        # -------------------------------------------------------------------------- 
        # apply diffusion forward
            t, _ = self.schedule_sampler.sample(x.shape[0], x.device)
            noise = torch.randn_like(x)
            x = self.diffusion.q_sample(x, t, noise=noise)

            if np.random.rand(1)[0] < 0.001:
                print('x_aug', x_aug[0].reshape(-1, C))
                print('x_gt', x_gt[0].reshape(-1, C))
                print('x_noisy', x[0].reshape(-1, C))
                print('t =', t[0])
        # -------------------------------------------------------------------------- 

        latent, cls_token, mask, ids_restore = self.forward_encoder(x_aug, x_orig=x_orig, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        
        loss_uni = token_uniformity_loss(latent.reshape(N, M, -1, latent.shape[-1])[:,0])
        #loss_uni = token_uniformity_loss(latent.reshape(N, M, -1, latent.shape[-1])[:,0], normalize=False)
        
        pred = self.forward_decoder(x, latent=latent, cls_token=cls_token, t=t, ids_restore=ids_restore)  # [NM, TP * VP, C]
        
        if torch.any(torch.isnan(pred)): print('Error! Nan in pred.')

        if self.diff_prediction == 'joint':
            loss_diff = self.forward_loss(x_gt, pred, mask, t)
        elif self.diff_prediction == 'noise':
            loss_diff = self.forward_loss(noise, pred, mask, t)
        elif self.diff_prediction == 'noise2joint':
            pred = self.diffusion._predict_xstart_from_eps(self.patchify(x), t, pred)
            loss_diff = self.forward_loss(x_gt, pred, mask, t)
        elif self.diff_prediction == 'v':
            v = self.diffusion.get_velocity(x_gt, t, noise=noise)
            loss_diff = self.forward_loss(v, pred, mask, t)
        else: 
            assert 0

        loss = loss_diff + self.lambda_loss_uni * loss_uni

        if np.random.rand(1)[0] < 0.01: print(f'loss = {loss.item()}, loss_diff = {loss_diff.item()}, loss_uni = {loss_uni.item()}, lambda = {self.lambda_loss_uni}, mask_ratio = {mask_ratio:.3f}, bs = {N}')

        return loss, pred, mask #mask: 0 is keep, 1 is remove

    def update_diffusion_sampler(self, epoch, total_epoch):
        print('Not update diffusion sampler')