import torch
import datasets, loss
import os
import argparse
from torch.utils.data import DataLoader
import models
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from bisect import bisect
import torchvision.transforms.functional as TF
from torch.nn.functional import affine_grid, grid_sample
import warnings

warnings.filterwarnings("ignore")
#-----------------------------------------new_models--------------------------------------------------#
#----------------------------------------------------------------------------------------------#
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


#---------------------------------------------CSWT_Unet-----------------------------------------------------#

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
class BigDataset_two(torch.utils.data.Dataset):
    def __init__(self, data_paths_1):
        self.data_memmaps = [np.load(path, mmap_mode='r+') for path in data_paths_1]
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

        up_img = up_img.transpose(2,0,1)
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
#----------------------------------------训练的类-----------------------------------------------------#
from Uformers import *
class Trainer:
    record = {"train_loss_d": [],"train_loss_g":[],"train_psnr": [],"train_ssim": [],"val_loss": [],"val_psnr":[],"val_ssim":[]}
    x_epoch = []

    def __init__(self,args):
        self.args = args
        self.device = args.device
        input_size = 128
        # arch = Uformer
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.gnet = Uformer(img_size=input_size, embed_dim=16, depths=depths,
                                    win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff',
                                    modulator=True,
                                    shift_flag=False)
        self.dnet = models.Discriminator()
        batch = self.args.batch
        # self.train_loader = DataLoader(BigDataset([self.args.train_data_path1,self.args.train_data_path2,self.args.train_data_path3,self.args.train_data_path4],[self.args.train_data_path5,self.args.train_data_path6,self.args.train_data_path7,self.args.train_data_path8]),batch_size=batch,shuffle=True,drop_last=True)
        self.train_loader = DataLoader(datasets.BigDataset(
            [self.args.train_data_path8,self.args.train_data_path9,self.args.train_data_path10,self.args.train_data_path11,
             self.args.train_data_path12,self.args.train_data_path13,self.args.train_data_path14,self.args.train_data_path15]),
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(datasets.BigDataset([self.args.val_data_path1]),
                                     batch_size=48, shuffle=False, drop_last=True)
        #生成器的损失
        self.criterion_g = loss.PerceptualLoss(self.device)
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
                self.epoch = param_dict["epoch"]
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
        self.optimizer_g = torch.optim.Adam(self.gnet.parameters(),lr=self.lr*0.1)
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
        for i, (index, img, up_label) in enumerate(self.train_loader):
            img = img.float()
            # down_label = down_label.float()
            up_label = up_label.float()
            img = img.to(self.device)
            up_label = up_label.to(self.device)
            # down_label = down_label.to(self.device)
            fake_up_img = self.gnet(img)
            loss_g = (self.criterion_g(fake_up_img, up_label, self.dnet(up_label)) + 2e-8 * self.regularization(
                fake_up_img))
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            # 判别器每隔一次进行一次迭代
            # if  epoch > 2:
            #     up_real_out = self.dnet(up_label.float())
            #     up_fake_out = self.dnet(fake_up_img.detach().float())
            #     down_real_out = self.dnet(down_label.float())
            #     # down_fake_out = self.dnet(fake_down_img.detach().float())
            #     loss_d = (self.criterion_d(up_real_out.to(torch.double), self.real_up_label.to(torch.double)
            #                                ) + self.criterion_d(up_fake_out.to(torch.double),
            #                                                     self.fake_up_label.to(torch.double)) +
            #               self.criterion_d(down_real_out.to(torch.double), self.real_down_label.to(torch.double)
            #                                ) + self.criterion_d(down_fake_out.to(torch.double),
            #                                                     self.fake_down_label.to(torch.double))) / 2
            #     self.optimizer_d.zero_grad()
            #     loss_d.backward()
            #     self.optimizer_d.step()
            #
            #     train_loss_d += loss_d.item()
            #     train_loss_all_d += loss_d.item()

            train_loss_g += loss_g.item()
            train_loss_all_g += loss_g.item()
            psnr += self.calculate_psnr(fake_up_img, up_label).item()
            ssim += self.calculate_ssim(fake_up_img, up_label).item()
            total += 1
            if (i + 1) % self.args.interval == 0:
                end = time.time()
                print(
                    "[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} dnet_loss:{:.5f} gnet_loss:{:.5f} psnr:{:.4f} ssim:{:.4f}".format(
                        epoch, (i + 1) * 100 / len(self.train_loader), end - start,
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
        return train_loss_all_d / (len(self.train_loader)), train_loss_all_g / (
            len(self.train_loader)), psnr / total, ssim / total

    # 验证主函数
    def val(self, epoch):
        self.gnet.eval()
        self.dnet.eval()
        print("Test start...")
        val_loss = 0.
        psnr = 0.
        total = 0
        ssim = 0.
        start = time.time()
        # fake_up_imgs = np.zeros((2000,3, 3, 128, 128))
        with torch.no_grad():
            for i, (index, img, up_label) in enumerate(self.val_loader):
                img = img.float()
                up_label = up_label.float()
                # fake_up_imgs[i, 0,:, :, :] = np.squeeze(up_label.cpu().numpy())
                # fake_up_imgs[i, 1,:, :, :] = np.squeeze(img.cpu().numpy())
                img = img.to(self.device)
                up_label = up_label.to(self.device)
                fake_up_img = self.gnet(img)
                # fake_up_imgs[i, 2,:, :, :] = np.squeeze(fake_up_img.cpu().numpy())
                loss = self.criterion_g(fake_up_img, up_label, self.dnet(up_label))

                val_loss += loss.item()
                psnr += self.calculate_psnr(fake_up_img, up_label).item()
                ssim += self.calculate_ssim(fake_up_img, up_label).item()
                total += 1

                if i >= 1999:
                    break
            mpsnr = psnr / total
            mssim = ssim / total
            end = time.time()
            print("Test finished!")
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} psnr:{:.4f} ssim:{:.4f}".format(
                epoch, end - start, val_loss / len(self.val_loader), mpsnr, mssim
            ))
            # 保存所有epoch中mpsnr指标最大的权重
            if mpsnr > self.best_psnr:
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
            # np.save("./val_fake_up_img.npy", fake_up_imgs)
        return val_loss / len(self.val_loader), mpsnr, mssim

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
        ax0.plot(self.x_epoch, self.record["train_loss_d"], "bo-", label="train_d")
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

def main(args):
    t = Trainer(args)
    #fig = plt.figure()
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss_d, train_loss_g, train_psnr ,train_ssim = t.train(epoch)
        val_loss, val_psnr ,val_ssim = t.val(epoch)
        #t.draw_curve(fig, epoch, train_loss_d, train_loss_g, train_psnr,train_ssim,val_loss, val_psnr,val_ssim)
        if (epoch + 1) % 10 == 0:
            t.lr_update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Galaxy Deblender Models")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train_data_path1", default="../train_data_npy/train_0.npy", type=str)
    parser.add_argument("--train_data_path2", default="../train_data_npy/train_1.npy", type=str)
    parser.add_argument("--train_data_path3", default="../train_data_npy/train_2.npy", type=str)
    parser.add_argument("--train_data_path4", default="../train_data_npy/train_3.npy", type=str)
    parser.add_argument("--train_data_path5", default="../train_data_npy/train_4.npy", type=str)
    parser.add_argument("--train_data_path6", default="../train_data_npy/train_5.npy", type=str)
    parser.add_argument("--train_data_path7", default="../train_data_npy/train_6.npy", type=str)
    parser.add_argument("--train_data_path8", default="../train_data_npy/train_7.npy", type=str)
    parser.add_argument("--train_data_path9", default="../new_train_data_npy/train_8.npy", type=str)
    parser.add_argument("--train_data_path10", default="../new_train_data_npy/train_9.npy", type=str)
    parser.add_argument("--train_data_path11", default="../new_train_data_npy/train_10.npy", type=str)
    parser.add_argument("--train_data_path12", default="../new_train_data_npy/train_11.npy", type=str)
    parser.add_argument("--train_data_path13", default="../new_train_data_npy/train_12.npy", type=str)
    parser.add_argument("--train_data_path14", default="../new_train_data_npy/train_13.npy", type=str)
    parser.add_argument("--train_data_path15", default="../new_train_data_npy/train_14.npy", type=str)
    parser.add_argument("--train_data_path16", default="../new_train_data_npy/train_15.npy", type=str)
    parser.add_argument("--val_data_path1", default="../val_data_npy/val.npy", type=str)
    parser.add_argument("--val_data_path2", default="../val_data_npy/val_1.npy", type=str)
    parser.add_argument("--val_data_path3", default="../val_data_npy/val_2.npy", type=str)
    parser.add_argument("--val_data_path4", default="../val_data_npy/val_3.npy", type=str)
    parser.add_argument("--val_data_path5", default="../val_data_npy/val_4.npy", type=str)
    parser.add_argument("--val_data_path6", default="../val_data_npy/val_5.npy", type=str)
    parser.add_argument("--val_data_path7", default="../val_data_npy/val_6.npy", type=str)
    # parser.add_argument("--val_data_path2", default="../val_data_npy/val_1.npy", type=str)
    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--save_path", default="./Uformer_up_weight.pt", type=str)
    parser.add_argument("--save_path1", default="./Uformer_up_weight1.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=48, type=int)
    args1 = parser.parse_args()
    main(args1)







