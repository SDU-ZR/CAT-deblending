import random

import torch
import datasets, loss
import os
import argparse

from PIL.Image import Image
from torch.utils.data import DataLoader

import model
from torch.autograd import Variable
from math import exp

from bisect import bisect

import warnings

from Galaxy_deblending.origin_band_train.data.DECaLS.evalute_or_deblending import dr2_style_rgb
from Uformers import Uformer
# from deblender import CAT_Unet
# from download_images_threaded import dr2_style_rgb

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


class Mlp(nn.Module):
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


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe

        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C


        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x
from einops import rearrange
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x
class Upsample_1(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_1, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        # x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x
#---------------------------------------------CSWT_Unet-----------------------------------------------------#
class CSWT_Unet(nn.Module):
    def __init__(self, img_size=128,
                 in_chans=3,
                 embed_dim=96,
                 depth=[2, 2, 6, 2],
                 split_size=[1,3, 5, 7],
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_chk=False):
        super(CSWT_Unet, self).__init__()
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 2, 1),
            Rearrange('b c h w -> b (h w) c', h=img_size // 2, w=img_size // 2),
            nn.LayerNorm(embed_dim)
        )
        curr_dim = embed_dim
        dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        self.encoder_level1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_encoder[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.encoder_level2 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_encoder[i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.encoder_level3 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_encoder[i], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.latent = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[3],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_encoder[i], norm_layer=norm_layer)
            for i in range(depth[3])])
        self.up4_3 = Upsample(curr_dim)
        self.reduce_chan_level3 = nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=1)
        curr_dim = curr_dim // 2
        self.decoder_level3 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.up3_2 = Upsample(curr_dim)
        self.reduce_chan_level2 = nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=1)
        curr_dim = curr_dim // 2
        self.decoder_level2 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.up2_1 = Upsample(curr_dim)
        self.reduce_chan_level1 = nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=1)
        curr_dim = curr_dim // 2
        self.decoder_level1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.up1_1 = Upsample_1(curr_dim)
        self.output = nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp_img):
        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.stage1_conv_embed(inp_img)
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(inp_enc_level1)

        inp_enc_level2 = self.merge1(out_enc_level1)
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2)

        inp_enc_level3 = self.merge2(out_enc_level2)
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3)

        inp_enc_level4 = self.merge3(out_enc_level3)
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent)

        inp_dec_level3 = self.up4_3(latent, H // 16, W // 16)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 8, w=W // 8).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c")
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 8,W // 8)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c")
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2)
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 4, W // 4)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        inp_dec_level1 = rearrange(inp_dec_level1, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec_level1 = rearrange(inp_dec_level1, "b c h w -> b (h w) c")
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1)
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up1_1(out_dec_level1,H // 2,W // 2)
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1

#-----------------------------------dataloader--------------------------------------------------------#
class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths_1,data_paths_2):
        self.data_memmaps = [np.load(path, mmap_mode='r+') for path in data_paths_1]
        self.target_memmaps = [np.load(path, mmap_mode='r+') for path in data_paths_2]
        self.start_indices = [0] * len(data_paths_1)
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
            
    def __len__(self):
        return self.data_count
    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        up_img = self.data_memmaps[memmap_index][index_in_memmap, :, :, :]
        down_img = self.target_memmaps[memmap_index][index_in_memmap, :, :, :]

        up_img = up_img.transpose(2,0,1)
        up_img = torch.from_numpy(up_img)
        down_img = down_img.transpose(2, 0, 1)
        down_img = torch.from_numpy(down_img)

        return index, up_img,down_img
# class BigDataset_two(torch.utils.data.Dataset):
#     def __init__(self, data_paths_1):
#         self.data_memmaps = [np.load(path, mmap_mode='r+') for path in data_paths_1]
#         self.start_indices = [0] * len(data_paths_1)
#         self.data_count = 0
#         for index, memmap in enumerate(self.data_memmaps):
#             self.start_indices[index] = self.data_count
#             self.data_count += memmap.shape[0]
#
#     def __len__(self):
#         return self.data_count
#
#     def __getitem__(self, index):
#         memmap_index = bisect(self.start_indices, index) - 1
#         index_in_memmap = index - self.start_indices[memmap_index]
#         up_img = self.data_memmaps[memmap_index][index_in_memmap, :, :, :]
#
#         up_img = up_img.transpose(2,0,1)
#         up_img = torch.from_numpy(up_img)
#
#         return index, up_img
class BigDataset_two(torch.utils.data.Dataset):
    def __init__(self, data_paths_1):
        # 将所有数据加载到内存中
        self.data = [np.load(path) for path in data_paths_1]
        self.start_indices = [0] * len(data_paths_1)
        self.data_count = 0
        for index, data in enumerate(self.data):
            self.start_indices[index] = self.data_count
            self.data_count += data.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        up_img = self.data[memmap_index][index_in_memmap, :, :, :]

        # 转置维度并转换为Torch张量
        up_img = up_img.transpose(2, 0, 1)
        up_img = torch.from_numpy(up_img)

        return index, up_img

#-------------------------------------------------计算SSIM---------------------------------------------#
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
def ssim(img1, img2, L=1):
    """
    计算两个图像的结构相似性指标（SSIM）

    参数：
    img1, img2: 两个输入图像，应为相同大小的灰度图像
    L: 图像的动态范围，默认为255

    返回值：
    ssim_index: SSIM值
    """
    # 1. 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    # 2. 计算方差
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    # 3. 计算协方差
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    # 4. SSIM计算参数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    # 5. 计算SSIM指数
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_index = numerator / denominator
    return ssim_index
class Trainer:
    record = {"train_loss_d": [],"train_loss_g":[],"train_psnr": [],"train_ssim": [],"val_loss": [],"val_psnr":[],"val_ssim":[]}
    x_epoch = []

    def __init__(self,args):
        self.args = args
        self.device = args.device
        input_size = 128
        # arch = Uformer
        depths = [2, 2, 4, 6, 4, 6, 4, 2, 2]
        self.gnet = Uformer(img_size=input_size, embed_dim=16, depths=depths,
                            win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
                            shift_flag=False)
        # self.gnet = GRL(
        #     upscale=4,
        #     img_size=128,
        #     window_size=8,
        #     depths=[4, 4, 4, 4],
        #     embed_dim=48,
        #     num_heads_window=[2,2,2,2],
        #     num_heads_stripe=[2,2,2,2],
        #     mlp_ratio=2,
        #     qkv_proj_type="linear",
        #     anchor_proj_type="avgpool",
        #     anchor_window_down_factor=2,
        #     out_proj_type="linear",
        #     conv_type="1conv",
        #     upsampler="None",
        #     local_connection=True,
        # )
        # self.gnet = model_gan.Generator()
        # self.gnet = MaskedAutoencoderViT(img_size=128, patch_size=16, in_chans=3,
        #          embed_dim=1024, depth=24, num_heads=16,
        #          decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        #          mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
        # self.gnet = CAT_Unet(
        # img_size= 128,
        # in_chans= 3,
        # depth= [4,6,6,8],
        # split_size_0= [4,4,4,4],
        # split_size_1= [0,0,0,0],
        # dim= 48,
        # num_heads= [2,2,4,8],
        # mlp_ratio= 4,
        # num_refinement_blocks= 4,
        # bias= False,
        # dual_pixel_task= False,
        # )
        """
        in_chans: 3
        img_size: 128
        split_size_0: [4, 4, 4, 4]
        split_size_1: [0, 0, 0, 0]
        depth: [4, 6, 6, 8]
        dim: 48
        num_heads: [2, 2, 4, 8]
        mlp_ratio: 4
        num_refinement_blocks: 4
        bias: False
        dual_pixel_task: False
        """
        self.dnet = model.Discriminator()
        batch = self.args.batch
        # self.train_loader = DataLoader(BigDataset([self.args.train_data_path1,self.args.train_data_path2,self.args.train_data_path3,self.args.train_data_path4],[self.args.train_data_path5,self.args.train_data_path6,self.args.train_data_path7,self.args.train_data_path8]),batch_size=batch,shuffle=True,drop_last=True)
        self.train_loader_up = DataLoader(BigDataset_two([self.args.train_data_path1,self.args.train_data_path2,self.args.train_data_path3,]),batch_size=batch, shuffle=True, drop_last=True)
        self.train_loader_down = DataLoader(BigDataset_two([self.args.train_data_path4,self.args.train_data_path5,self.args.train_data_path6,self.args.train_data_path7,self.args.train_data_path8]),batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader_up = DataLoader(BigDataset_two([self.args.val_data_path1,self.args.val_data_path2,self.args.val_data_path3,]),
                                     batch_size=batch, shuffle=False, drop_last=True)
        self.val_loader_down = DataLoader(BigDataset_two([self.args.val_data_path4,self.args.val_data_path5,self.args.val_data_path6,]),
                                        batch_size=batch, shuffle=False, drop_last=True)
        #生成器的损失
        self.criterion_g = loss.PerceptualLoss(self.device)
        self.criterion_g_2 = loss.PerceptualLoss(self.device)
        # 正则化损失（Regularization Loss）是一种在神经网络中用于降低过拟合的技术
        # 其目的是在损失函数中增加一个惩罚项，以避免模型过度拟合训练数据，从而提高泛化能力
        self.regularization = loss.RegularizationLoss()
        #二维交叉熵做损失函数，作为判别器损失
        self.criterion_d = torch.nn.BCELoss()
        self.epoch = 0
        self.lr = 1e-3
        # 峰值信噪比
        self.best_psnr = 0.
        # 保存最佳结构相似性系数
        self.best_ssim = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params,star training...")

            else:
                param_dict = torch.load(self.args.save_path)
                self.epoch = param_dict["epoch"] + 1
                self.lr = 0.0001
                self.dnet.load_state_dict(param_dict["dnet_dict"])
                self.gnet.load_state_dict(param_dict["gnet_dict"])
                self.best_psnr = param_dict["best_psnr"]
                self.best_ssim = param_dict["best_ssim"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_psnr]: {}  [best_ssim]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                 self.best_psnr,self.best_ssim))

        self.dnet.to(self.device)
        self.gnet.to(self.device)
        self.optimizer_d = torch.optim.Adam(self.dnet.parameters(),lr=self.lr)
        self.optimizer_g = torch.optim.Adam(self.gnet.parameters(),lr=self.lr)
        self.real_up_label = torch.ones([batch, 1]).to(self.device)
        self.fake_down_label = torch.zeros([batch, 1]).to(self.device)
        self.fake_up_label = torch.zeros([batch, 1]).to(self.device)
        self.real_down_label = torch.ones([batch, 1]).to(self.device)
    #------------------两个静态方法计算结构相似性系数和高值信噪比---------------------------#
    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    @staticmethod
    def calculate_ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)
    #------------------------------------------------------------------------------#
    # 训练主函数
    def train(self,epoch):
        self.dnet.train()
        self.gnet.train()
        train_loss_d = 0.
        train_loss_g = 0.
        train_loss_all_d = 0.
        train_loss_all_g = 0.
        psnr = 0.
        total = 0
        ssim = 0.
        start = time.time()
        print("Start epoch: {}".format(epoch))
        for i, ((index, up_label),(index,down_label)) in enumerate(zip(self.train_loader_up,self.train_loader_down)):
            down_label = down_label.float()
            up_label = up_label.float().clamp(-1,2)
            img = up_label + down_label
            # img = img.clamp(-1,1)
            # img = torch.tanh(torch.arcsinh(img))
            img = img.clamp(-1,2).to(self.device)
            up_label = up_label.to(self.device)
            # up_label = torch.tanh(torch.arcsinh(up_label))
            # fake_up_img = self.gnet(img)
            fake_up_img = self.gnet(img)
            fake_up_img = fake_up_img.float().to("cuda")
            if epoch == 1000:
                loss_g = self.criterion_g(fake_up_img, up_label)
            else:
                loss_g = self.criterion_g_2(fake_up_img, up_label)
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()
            train_loss_g += loss_g.item()
            train_loss_all_g += loss_g.item()
            psnr += self.calculate_psnr(fake_up_img, up_label).item()
            ssim += self.calculate_ssim(fake_up_img, up_label).item()
            total += 1
            if (i + 1) % self.args.interval == 0:
                end = time.time()
                print(
                    "[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} dnet_loss:{:.5f} gnet_loss:{:.10f} psnr:{:.4f} ssim:{:.10f}".format(
                        epoch, (i + 1) * 100 / len(self.train_loader_up), end - start,
                               train_loss_d / self.args.interval,
                               train_loss_g / self.args.interval, psnr / total, ssim / total
                    ))
                train_loss_d = 0.
                train_loss_g = 0.

        # 每个epoch结束都保存，应该是为了训练意外终止时能够从记录的终止点进行训练
        print("Save params to {}".format(self.args.save_path))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "dnet_dict": self.dnet.state_dict(),
            "gnet_dict": self.gnet.state_dict()
        }
        torch.save(param_dict, self.args.save_path)
        return train_loss_all_d/(len(self.train_loader_up)), train_loss_all_g/(len(self.train_loader_up)), psnr/total,ssim/total

    def lupton(self,resize_image):
        _scales = dict(
            g=(2, 0.008),
            r=(1, 0.014),
            z=(0, 0.019))
        _mnmx = (-0.5, 300)

        rgbimg = dr2_style_rgb(
            (resize_image[0, :, :], resize_image[1, :, :], resize_image[2, :, :]),
            'grz',
            mnmx=_mnmx,
            arcsinh=1.,
            scales=_scales,
            desaturate=True)
        native_pil_image = Image.fromarray(np.uint8(rgbimg * 255.), mode='RGB')
        return native_pil_image

    def show(self,img,fake_up_img,up_label):
        import matplotlib.pyplot as plt

        # Assume img and fake_up_img are batches of images from the model output, each with shape (batch_size, channels, height, width)
        # Here we select the first image in the batch (index 0) for comparison

        # Extract the first image in the batch for both original and generated images
        index = random.randint(0,2)
        img = img[index].cpu().numpy() # Convert from (C, H, W) to (H, W, C) if needed
        fake_up_img = fake_up_img[index].cpu().numpy()
        up_label = up_label[index].cpu().numpy()


        img_first = self.lupton(img)
        up_label_first = self.lupton(up_label)
        fake_up_img_first = self.lupton(fake_up_img)

        # Plot the original and generated images side by side for comparison
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))  # 2行1列
        ax1.imshow(up_label_first)
        ax1.text(2., 10, r'original', color='#FFFFFF', fontsize=18)
        # ax1.text(4.3, 120, '{:4d}'.format(i), color='#FFFFFF',
        #          fontsize=12)
        ax2.imshow(img_first)
        ax2.text(2, 10, r'blended', color='#FFFFFF', fontsize=18)
        # ax2.text(4.3, 120, '{:4d}'.format(i), color='#FFFFFF',
        #          fontsize=12)
        ax3.imshow(fake_up_img_first)
        ax3.text(2, 10, r'after_restored', color='#FFFFFF', fontsize=18)
        ax3.text(2, 80, r'ssim: {:.4f}'.format(ssim(fake_up_img, up_label)), color='#FFFFFF', fontsize=18)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        plt.tight_layout()
        # plt.savefig("./picture/test_{}.png".format(i))
        # print("-------------------------------{}-----------------------".format(i))
        plt.show()

    # 验证主函数
    def val(self, epoch):
        self.gnet.eval()
        self.dnet.eval()
        # fake_up_imgs = np.zeros((10000,4,3,128, 128))
        print("Test start...")
        val_loss = 0.
        psnr = 0.
        total = 0
        ssim = 0.
        start = time.time()
        with torch.no_grad():
            for i, ((index, up_label), (index, down_label)) in enumerate(zip(self.val_loader_up, self.val_loader_down)):
                # fake_up_imgs[i, 0, :, :, :] = np.squeeze(up_label.cpu().numpy())
                # fake_up_imgs[i, 1, :, :, :] = np.squeeze(down_label.cpu().numpy())
                down_label = down_label.float()
                up_label = up_label.float().clamp(-1,2)
                img = up_label + down_label

                # fake_up_imgs[i, 2, :, :, :] = np.squeeze(img.cpu().numpy())
                # img = img.clamp(-1, 1)
                # img = torch.tanh(torch.arcsinh(img))
                img = img.clamp(-1,2).to(self.device)
                up_label = up_label.to(self.device)
                # up_label = torch.tanh(torch.arcsinh(up_label))
                up_label = up_label.to(self.device)
                fake_up_img = self.gnet(img)

                if i == 0:
                    self.show(img,fake_up_img,up_label)
                # fake_up_img = self.gnet(fake_up_img)
                # fake_up_img = self.gnet(fake_up_img)

                loss = self.criterion_g_2(fake_up_img, up_label)

                val_loss += loss.item()
                psnr += self.calculate_psnr(fake_up_img, up_label).item()
                ssim += self.calculate_ssim(fake_up_img, up_label).item()
                # fake_up_img = torch.sinh(torch.arctanh(fake_up_img))
                total += 1
                # fake_up_imgs[i,3,:,:,:] = np.squeeze(fake_up_img.cpu().numpy())
                # print("        完成第{}个测试    ".format(i))
                # if i == 9999:
                #     break

            mpsnr = psnr / total
            mssim = ssim / total
            end = time.time()
            print("Test finished!")
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} psnr:{:.4f} ssim:{:.4f}".format(
                epoch, end - start, val_loss / len(self.val_loader_up), mpsnr, mssim
            ))
            # 保存所有epoch中mpsnr指标最大的权重
            if mssim > self.best_ssim:
                self.best_psnr = mpsnr
                self.best_ssim = mssim
                print("Save params to {}".format(self.args.save_path1))
                param_dict = {
                    "epoch": epoch,
                    "lr": self.lr,
                    "best_psnr": self.best_psnr,
                    "best_ssim": self.best_ssim,
                    "gnet_dict": self.gnet.state_dict(),
                    "dnet_dict": self.dnet.state_dict()
                }
                torch.save(param_dict, self.args.save_path1)
            # np.save("./val_fake_up_img.npy",fake_up_imgs)
        return val_loss/len(self.val_loader_up), mpsnr,mssim

    def draw_curve(self, fig, epoch, train_loss_d, train_loss_g, train_psnr, train_ssim,val_loss, val_psnr,val_ssim):
        ax0 = fig.add_subplot(131, title="loss")
        ax1 = fig.add_subplot(132, title="psnr&ssim")
        ax2 = fig.add_subplot(133, title="psnr&ssim")
        self.record["train_loss_d"].append(train_loss_d)
        self.record["train_loss_g"].append(train_loss_g)
        self.record["train_psnr"].append(train_psnr)
        self.record["train_ssim"].append(train_ssim)
        self.record["val_loss"].append(val_loss)
        self.record["val_psnr"].append(val_psnr)
        self.record["val_ssim"].append(val_ssim)
        self.x_epoch.append(epoch)
        # ax0.plot(self.x_epoch, self.record["train_loss_d"], "bo-", label="train_d")
        ax0.plot(self.x_epoch, self.record["train_loss_g"], "go-", label="train_g")
        ax0.plot(self.x_epoch, self.record["val_loss"], "ro-", label="val_g")
        ax1.plot(self.x_epoch, self.record["train_psnr"], "bo-", label="train")
        ax2.plot(self.x_epoch, self.record["train_ssim"], "bo-", label="train")
        ax1.plot(self.x_epoch, self.record["val_psnr"], "ro-", label="val")
        ax2.plot(self.x_epoch, self.record["val_ssim"], "ro-", label="val")

        if epoch == 0:
            ax0.legend()
            ax1.legend()
            ax2.legend()
        fig.savefig(r"./train_fig/train_{}.jpg".format(epoch))

    def lr_update(self):
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = self.lr * 0.1
        self.lr = self.optimizer_d.param_groups[0]["lr"]
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.lr
        print("===============================================")
        print("Learning rate has adjusted to {}".format(self.lr))


def show(epoch,train_loss,val_loss,train_ssim,val_ssim,train_psnr,val_psnr):
    import matplotlib.pyplot as plt
    # Assuming that user will provide the six arrays (each of 100 elements)
    # Placeholder data for demonstration, replace with actual arrays when available
    import numpy as np

    # Placeholder data - 100 values for each metric across epochs
    epochs = np.arange(1, epoch)

    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Remove white borders around the figure
    fig.subplots_adjust(wspace=0.3, hspace=0)

    # Plot train_loss and val_loss
    axes[0].plot(epochs, train_loss, label='Train Loss', color='#1f77b4')
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss over Epochs')

    # Plot train_ssim and val_ssim
    axes[1].plot(epochs, train_ssim, label='Train SSIM', color='#2ca02c')
    axes[1].plot(epochs, val_ssim, label='Validation SSIM', color='#d62728')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('SSIM')
    axes[1].legend()
    axes[1].set_title('SSIM over Epochs')

    # Plot train_psnr and val_psnr
    axes[2].plot(epochs, train_psnr, label='Train PSNR', color='#9467bd')
    axes[2].plot(epochs, val_psnr, label='Validation PSNR', color='#8c564b')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('PSNR')
    axes[2].legend()
    axes[2].set_title('PSNR over Epochs')

    # Apply tight layout to reduce white spaces around the subplots
    plt.tight_layout()
    plt.savefig(r"./train_fig/train_{}.jpg".format(epoch[-1]))


    # Show the plot
    plt.show()


def main(args):
    t = Trainer(args)
    train_loss_all = []
    val_loss_all = []
    train_ssim_all = []
    val_ssim_all = []
    train_psnr_all = []
    val_psnr_all = []

    pre = t.epoch
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss_d, train_loss_g, train_psnr ,train_ssim = t.train(epoch)
        val_loss, val_psnr ,val_ssim = t.val(epoch)

        epochs = np.arange(pre, epoch + 1)
        train_loss_all.append(train_loss_g)
        train_psnr_all.append(train_psnr)
        train_ssim_all.append(train_ssim)
        val_loss_all.append(val_loss)
        val_psnr_all.append(val_psnr)
        val_ssim_all.append(val_ssim)

        # if epoch % 10 == 0:
        #     show(epochs, train_loss_all, val_loss_all, train_ssim_all, val_ssim_all, train_psnr_all, val_psnr_all)
        # t.draw_curve(fig, epoch, train_loss_d, train_loss_g, train_psnr,train_ssim,val_loss, val_psnr,val_ssim)
        if (epoch + 1) % 5 == 0:
            t.lr_update()
        if epoch == 30:
            print("            ---------------------训练暂时停止--------------------")
            print("                                                               ")
            break
    # 将数据转换为 NumPy 数组并保存
    np.savez('training_results.npz',
             train_loss=np.array(train_loss_all),
             train_psnr=np.array(train_psnr_all),
             train_ssim=np.array(train_ssim_all),
             val_loss=np.array(val_loss_all),
             val_psnr=np.array(val_psnr_all),
             val_ssim=np.array(val_ssim_all))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Galaxy Deblender Models")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train_data_path1", default="./data/up_data_1.npy", type=str)
    parser.add_argument("--train_data_path2", default="./data/up_data_2.npy", type=str)
    parser.add_argument("--train_data_path3", default="./data/up_data_3.npy", type=str)
    parser.add_argument("--train_data_path4", default="./data/bias_data_1.npy", type=str)
    parser.add_argument("--train_data_path5", default="./data/bias_data_2_1.npy", type=str)
    parser.add_argument("--train_data_path6", default="./data/bias_data_2_2.npy", type=str)
    parser.add_argument("--train_data_path7", default="./data/bias_data_3_1.npy", type=str)
    parser.add_argument("--train_data_path8", default="./data/bias_data_3_2.npy", type=str)

    parser.add_argument("--val_data_path1", default="./data/up_val_data.npy", type=str)
    parser.add_argument("--val_data_path2", default="./data/up_val_data_2.npy", type=str)
    parser.add_argument("--val_data_path3", default="./data/up_val_data_3.npy", type=str)
    parser.add_argument("--val_data_path4", default="./data/bias_val_data_1.npy", type=str)
    parser.add_argument("--val_data_path5", default="./data/bias_val_data_2.npy", type=str)
    parser.add_argument("--val_data_path6", default="./data/bias_val_data_4.npy", type=str)

    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    # parser.add_argument("--save_path", default="./Deblending_weight_uformer32.pt", type=str)
    # parser.add_argument("--save_path1", default="./Deblending_weight_uformer32_1.pt", type=str)
    parser.add_argument("--save_path", default="./Deblending_weight_uformer_decals.pt", type=str)
    parser.add_argument("--save_path1", default="./Deblending_weight_uformer_decals.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=64,type=int)
    args1 = parser.parse_args()
    main(args1)
