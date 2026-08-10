import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import copy
from .drop import DropPath
import sys
import numpy as np
from functools import partial
# diffusion
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import UniformSampler, MaskedDiffusionSampler, SNRSampler
from guided_diffusion.script_util import create_gaussian_diffusion

from .util import *
from .ose_memory import OSEMemory

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


class MomentumSkeletonEncoder(nn.Module):
    def __init__(self, online_model):
        super().__init__()
        self.joints_embed = copy.deepcopy(online_model.joints_embed)
        self.blocks = copy.deepcopy(online_model.blocks)
        self.norm = copy.deepcopy(online_model.norm)
        self.temp_embed = nn.Parameter(online_model.temp_embed.detach().clone())
        self.pos_embed = nn.Parameter(online_model.pos_embed.detach().clone())
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_from(self, online_model, momentum):
        online_parameters = list(online_model.joints_embed.parameters())
        online_parameters += list(online_model.blocks.parameters())
        online_parameters += list(online_model.norm.parameters())
        online_parameters += [online_model.temp_embed, online_model.pos_embed]
        momentum_parameters = list(self.joints_embed.parameters())
        momentum_parameters += list(self.blocks.parameters())
        momentum_parameters += list(self.norm.parameters())
        momentum_parameters += [self.temp_embed, self.pos_embed]
        for online, target in zip(online_parameters, momentum_parameters):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def forward(self, x, mask_ratio):
        x = self.joints_embed(x)
        batch_size, temporal_patches, joint_patches, dim = x.shape
        x = x + self.pos_embed[:, :, :joint_patches, :] + self.temp_embed[:, :temporal_patches, :, :]
        x = x.reshape(batch_size, temporal_patches * joint_patches, dim)
        keep = round(x.shape[1] * (1.0 - mask_ratio))
        ids = torch.argsort(torch.rand(batch_size, x.shape[1], device=x.device), dim=1)[:, :keep]
        x = torch.gather(x, 1, ids.unsqueeze(-1).expand(-1, -1, dim))
        for block in self.blocks:
            x = block(x)
        return F.normalize(self.norm(x).mean(dim=1), dim=-1)

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
        self.ose_memory = None
        self.ose_momentum_encoder = None
        self.ose_momentum = None
        self.ose_start_epoch = None
        self.lambda_ose = None
        self.self_prob_start = None
        self.self_prob_end = None
        self.peer_prob_start = None
        self.peer_prob_end = None

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

    def initialize_ose(
        self,
        exemplar_mapping,
        ose_momentum,
        ose_queue_size,
        ose_start_epoch,
        ose_topk,
        ose_alpha,
        ose_tau_s,
        ose_tau_t,
        ose_assignment_confidence,
        lambda_ose,
        self_prob_start,
        self_prob_end,
        peer_prob_start,
        peer_prob_end,
    ):
        class_ids = sorted(int(class_id) for class_id in exemplar_mapping)
        exemplar_ids = [int(exemplar_mapping[str(class_id)] if str(class_id) in exemplar_mapping else exemplar_mapping[class_id]) for class_id in class_ids]
        self.ose_memory = OSEMemory(
            feature_dim=self.dim_feat,
            class_ids=class_ids,
            exemplar_ids=exemplar_ids,
            queue_size=ose_queue_size,
            topk=ose_topk,
            alpha=ose_alpha,
            tau_s=ose_tau_s,
            tau_t=ose_tau_t,
            assignment_confidence=ose_assignment_confidence,
        )
        self.ose_momentum_encoder = MomentumSkeletonEncoder(self)
        self.register_buffer('ose_exemplars', torch.empty(0), persistent=False)
        self.ose_momentum = float(ose_momentum)
        self.ose_start_epoch = int(ose_start_epoch)
        self.lambda_ose = float(lambda_ose)
        self.self_prob_start = float(self_prob_start)
        self.self_prob_end = float(self_prob_end)
        self.peer_prob_start = float(peer_prob_start)
        self.peer_prob_end = float(peer_prob_end)

    def _sequence_layout(self, skeleton):
        if self.one_person:
            skeleton = skeleton[..., :1]
        batch_size, channels, frames, joints, people = skeleton.shape
        if not self.one_person:
            raise ValueError('OSE identities are sequence-level; set one_person=True')
        return skeleton.permute(0, 4, 2, 3, 1).contiguous().view(
            batch_size * people, frames, joints, channels
        )

    def _normalize_sequence(self, skeleton, center=None):
        skeleton = skeleton.clone()
        if self.self_shift:
            if center is None:
                center = skeleton.mean(dim=[1, 2], keepdim=True)
            skeleton = skeleton - center
        for channel in range(skeleton.shape[-1]):
            skeleton[..., channel] = (
                skeleton[..., channel] - self.input_mean[channel]
            ) / self.input_std[channel]
        return skeleton

    @torch.no_grad()
    def _prepare_source(self, source, source_aug):
        source = self._sequence_layout(source)
        source_aug = self._sequence_layout(source_aug)
        center = source.mean(dim=[1, 2], keepdim=True) if self.self_shift else None
        return (
            self._normalize_sequence(source, center=center),
            self._normalize_sequence(source_aug, center=center),
        )

    @torch.no_grad()
    def _prepare_target(self, target):
        target = self._sequence_layout(target)
        center = target.mean(dim=[1, 2], keepdim=True) if self.self_shift else None
        return self._normalize_sequence(target, center=center)

    @torch.no_grad()
    def update_momentum_encoder(self):
        self.ose_momentum_encoder.update_from(self, self.ose_momentum)

    @torch.no_grad()
    def reset_momentum_encoder(self):
        self.ose_momentum_encoder.update_from(self, 0.0)

    @torch.no_grad()
    def enqueue_ose(self, features, source_ids):
        self.ose_memory.enqueue(features, source_ids)

    @torch.no_grad()
    def set_ose_exemplars(self, exemplar_samples):
        if exemplar_samples.shape[0] != self.ose_memory.class_ids.numel():
            raise ValueError(
                'Exemplar sample count must match the exemplar mapping')
        self.ose_exemplars = exemplar_samples.detach()

    @torch.no_grad()
    def refresh_ose(self, exemplar_samples, mask_ratio):
        exemplar_samples = self._prepare_target(exemplar_samples)
        self.ose_momentum_encoder.eval()
        exemplar_features = self.ose_momentum_encoder(exemplar_samples, mask_ratio)
        return self.ose_memory.refresh(exemplar_features)

    @torch.no_grad()
    def ose_metrics(self):
        return self.ose_memory.metrics()
        
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

    def build_global_local_condition(self, latent, cls_token, ids_restore):
        temporal_patches = self.joints_embed.t_grid_size
        joint_patches = self.joints_embed.grid_size
        dim = latent.shape[-1]
        condition = torch.cat([
            latent,
            cls_token.repeat(1, temporal_patches * joint_patches - latent.shape[1], 1),
        ], dim=1)
        return torch.gather(
            condition,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
        )

    def forward_decoder(self, x, *, z, t):
        NM = x.shape[0]
        TP = self.joints_embed.t_grid_size
        VP = self.joints_embed.grid_size
        C = self.dim_feat

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

    def forward(
        self,
        source,
        source_aug,
        peer,
        source_ids,
        has_peer,
        epoch,
        total_epochs,
        mask_ratio=0.90,
        motion_aware_tau=-1,
    ):
        if motion_aware_tau > 0:
            raise ValueError('OSE peer diffusion requires random masking (motion_aware_tau <= 0)')
        if self.ose_memory is None:
            raise RuntimeError('initialize_ose must be called before OSE pretraining')

        with torch.no_grad():
            source, source_aug = self._prepare_source(source, source_aug)
            peer = self._prepare_target(peer)
            source_orig = source.clone()

        latent, cls_token, mask, ids_restore = self.forward_encoder(
            source_aug,
            x_orig=source_orig,
            mask_ratio=mask_ratio,
            motion_aware_tau=motion_aware_tau,
        )
        with torch.no_grad():
            self.ose_momentum_encoder.eval()
            teacher_features = self.ose_momentum_encoder(source, mask_ratio)

        ose_active = int(epoch) >= self.ose_start_epoch
        zero = cls_token.new_zeros(())
        confidence = cls_token.new_zeros(source.shape[0])
        ose_losses = {
            'proto': zero,
            'align': zero,
            'dispersion': zero,
            'target_entropy': zero,
            'align_kl': zero,
        }
        if ose_active:
            if self.ose_exemplars.numel() == 0:
                raise RuntimeError('OSE exemplar samples have not been installed')
            global_features = F.normalize(cls_token.squeeze(1), dim=-1)
            exemplar_samples = self._prepare_target(self.ose_exemplars)
            _, exemplar_cls, _, _ = self.forward_encoder(
                exemplar_samples,
                x_orig=exemplar_samples,
                mask_ratio=mask_ratio,
                motion_aware_tau=-1,
            )
            exemplar_features = F.normalize(exemplar_cls.squeeze(1), dim=-1)
            ose_losses = self.ose_memory.prototype_loss(
                global_features, teacher_features, exemplar_features)
            confidence = ose_losses['confidence']
            progress = (
                float(epoch - self.ose_start_epoch)
                / max(int(total_epochs) - self.ose_start_epoch - 1, 1)
            )
            p_peer = self.peer_prob_start + (
                self.peer_prob_end - self.peer_prob_start) * progress
            p_self = self.self_prob_start + (
                self.self_prob_end - self.self_prob_start) * progress
            requested_peer = (
                torch.rand(source.shape[0], device=source.device) < p_peer)
            confident = confidence >= self.ose_memory.assignment_confidence
            use_peer = requested_peer & has_peer.bool() & confident
        else:
            p_peer = 0.0
            p_self = 1.0
            confident = torch.zeros_like(has_peer, dtype=torch.bool)
            use_peer = torch.zeros_like(has_peer, dtype=torch.bool)

        with torch.no_grad():
            target = torch.where(use_peer[:, None, None, None], peer, source)
            target_gt = target.clone()
            t, _ = self.schedule_sampler.sample(target.shape[0], target.device)
            noise = torch.randn_like(target)
            noisy_target = self.diffusion.q_sample(target, t, noise=noise)

        loss_uniformity = token_uniformity_loss(latent)
        z_self = self.build_global_local_condition(latent, cls_token, ids_restore)
        z_peer = cls_token.repeat(1, z_self.shape[1], 1)
        z = torch.where(use_peer[:, None, None], z_peer, z_self)
        pred = self.forward_decoder(noisy_target, z=z, t=t)

        if self.diff_prediction == 'joint':
            loss_diff = self.forward_loss(target_gt, pred, mask, t)
        elif self.diff_prediction == 'noise':
            loss_diff = self.forward_loss(noise, pred, mask, t)
        elif self.diff_prediction == 'noise2joint':
            pred = self.diffusion._predict_xstart_from_eps(self.patchify(noisy_target), t, pred)
            loss_diff = self.forward_loss(target_gt, pred, mask, t)
        elif self.diff_prediction == 'v':
            velocity = self.diffusion.get_velocity(target_gt, t, noise=noise)
            loss_diff = self.forward_loss(velocity, pred, mask, t)
        else:
            raise ValueError('Unsupported diffusion prediction target: %s' % self.diff_prediction)

        loss = loss_diff + self.lambda_loss_uni * loss_uniformity
        if ose_active:
            loss = loss + self.lambda_ose * ose_losses['proto']
        aux = {
            'loss_diff': loss_diff.detach(),
            'loss_uniformity': loss_uniformity.detach(),
            'loss_ose_proto': ose_losses['proto'].detach(),
            'loss_ose_align': ose_losses['align'].detach(),
            'loss_ose_dispersion': ose_losses['dispersion'].detach(),
            'ose_target_entropy': ose_losses['target_entropy'].detach(),
            'ose_align_kl': ose_losses['align_kl'].detach(),
            'ose_active': torch.tensor(float(ose_active), device=source.device),
            'p_self_planned': torch.tensor(p_self, device=source.device),
            'p_peer_planned': torch.tensor(p_peer, device=source.device),
            'self_fraction_effective': (~use_peer).float().mean().detach(),
            'peer_fraction_effective': use_peer.float().mean().detach(),
            'ose_confidence': confidence.mean().detach(),
            'above_confidence_fraction': confident.float().mean().detach(),
        }
        # Return the realized routing decision for label-only diagnostics in
        # the training engine. Ground-truth labels never enter this forward.
        return loss, pred, mask, aux, teacher_features.detach(), use_peer.detach()

    def update_diffusion_sampler(self, epoch, total_epoch):
        print('Not update diffusion sampler')
