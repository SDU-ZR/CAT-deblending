from bisect import bisect
from scipy.interpolate import interp1d
import torch
import datasets, loss
import os
import argparse
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
#-------------------------------------------new model-------------------------------------------------#
import torch
import torch.nn as nn

from timm.models.layers import DropPath
from einops import rearrange

from Galaxy_deblending.rgb_train.deblender_train import CAT_Unet


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


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


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention_axial(nn.Module):
    """ Axial Rectangle-Window (axial-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, -1 is Full Attention, 0 is V-Rwin, 1 is H-Rwin.
        split_size (int): Height or Width of the regular rectangle window, the other is H or W (axial-Rwin).
        dim_out (int | None): The dimension of the attention output, if None dim_out is dim. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weights. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.tmp_H = H_sp
        self.tmp_W = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        # the side of axial rectangle window changes with input
        if self.resolution != H or self.resolution != W:
            if self.idx == -1:
                H_sp, W_sp = H, W
            elif self.idx == 0:
                H_sp, W_sp = H, self.split_size
            elif self.idx == 1:
                W_sp, H_sp = W, self.split_size
            else:
                print ("ERROR MODE", self.idx)
                exit(0)
            self.H_sp = H_sp
            self.W_sp = W_sp
        else:
            self.H_sp = self.tmp_H
            self.W_sp = self.tmp_W

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=attn.device)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij')) # for pytorch >= 1.10
            # biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w])) # for pytorch < 1.10
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp, device=attn.device)
            coords_w = torch.arange(self.W_sp, device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # for pytorch >= 1.10
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # for pytorch < 1.10
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)

            pos = self.pos(biases)
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = x.transpose(1, 2).contiguous().reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class CATB_axial(nn.Module):
    """ Axial Cross Aggregation Transformer Block.
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (int): Height or Width of the axial rectangle window, the other is H or W (axial-Rwin).
        shift_size (int): Shift size for axial-Rwin.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, reso, num_heads,
                 split_size=7, shift_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        assert 0 <= self.shift_size < self.split_size, "shift_size must in 0-split_size"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                    Attention_axial(
                        dim//2, resolution=self.patches_resolution, idx = i,
                        split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                    for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, self.split_size, 1))
        img_mask_1 = torch.zeros((1, self.split_size, W, 1))
        slices = (slice(-self.split_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            img_mask_0[:, :, s, :] = cnt
            img_mask_1[:, s, :, :] = cnt
            cnt += 1

        # calculate mask for V-Shift
        img_mask_0 = img_mask_0.view(1, H // H, H, self.split_size // self.split_size, self.split_size, 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, self.split_size, 1)
        mask_windows_0 = img_mask_0.view(-1, H * self.split_size)
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
        num_v = W // self.split_size
        attn_mask_0_la = torch.zeros((num_v,H * self.split_size,H * self.split_size))
        attn_mask_0_la[-1] = attn_mask_0

        # calculate mask for H-Shift
        img_mask_1 = img_mask_1.view(1, self.split_size // self.split_size, self.split_size, W // W, W, 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size, W, 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size * W)
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        num_h = H // self.split_size
        attn_mask_1_la = torch.zeros((num_h,W * self.split_size,W * self.split_size))
        attn_mask_1_la[-1] = attn_mask_1

        return attn_mask_0_la, attn_mask_1_la

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H , W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, C)
            # V-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=-self.shift_size, dims=3)
            qkv_0 = qkv_0.view(3, B, L, C//2)
            # H-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=-self.shift_size, dims=2)
            qkv_1 = qkv_1.view(3, B, L, C//2)

            if self.patches_resolution != H or self.patches_resolution != W:
                mask_tmp = self.calculate_mask(H, W)
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=mask_tmp[0].to(x.device))
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=mask_tmp[1].to(x.device))

            else:
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=self.attn_mask_0)
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=self.shift_size, dims=2)
            x2 = torch.roll(x2_shift, shifts=self.shift_size, dims=1)
            x1 = x1.view(B, L, C//2).contiguous()
            x2 = x2.view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], H, W).view(B, L, C//2).contiguous()
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], H, W).view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # Locality Complementary Module
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        attened_x = attened_x + lcm

        attened_x = self.proj(attened_x)
        attened_x = self.proj_drop(attened_x)

        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


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

def convert_image(down_label):
    fake_down_img = torch.squeeze(down_label)
    fake_down_img = fake_down_img.clamp(0.0, 1.0)
    fake_down_img = fake_down_img.permute(1, 2, 0)
    fake_down_img = np.array(fake_down_img.cpu().detach())
    return fake_down_img



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def make_plot(blended, true_x, true_y, gan_x, gan_y, ssim_x, ssim_y):
    """
    Plots paneled figure of preblended, blended and deblended galaxies.
    """
    fig = plt.Figure()
    ssim_x = np.around(ssim_x, decimals=2)
    ssim_y = np.around(ssim_y, decimals=2)

    gs = GridSpec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0:2, 1:3])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 3])

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axis('off')

    ax1.imshow(true_x)
    ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
    ax2.imshow(true_y)
    ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
    ax3.imshow(blended)
    ax3.text(1.3, 4.4, r'Blended', color='#FFFFFF')
    ax4.imshow(gan_x)
    ax4.text(3., 10., r'Deblended 1', color='#FFFFFF')
    ax4.text(3., 75., str(ssim_x) + ' dB', color='#FFFFFF')  #
    ax5.imshow(gan_y)
    ax5.text(3., 10., r'Deblended 2', color='#FFFFFF')
    ax5.text(3., 75., str(ssim_y) + ' dB', color='#FFFFFF')  #

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.06, hspace=-0.42)
    plt.show()
    """
    filename = os.path.join(savedir, 'test.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    """

#----------------------------------------------------dataloader-------------------------------------#
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
#-------------------------------------------显示结果-------------------------------------------------#
def psnr_show(psnr_values):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    # 生成5000个近似符合正态分布的PSNR值


    # 将PSNR值分为小区间
    num_bins = int((45 - 18) / 0.5)
    bins = [i for i in np.arange(18, 45, 0.5)]

    # 统计每个区间的数量
    hist, bins = np.histogram(psnr_values, bins=bins, density=True)

    # 计算中位数和均值
    median_psnr = np.median(psnr_values)
    mean_psnr = np.mean(psnr_values)
    print(mean_psnr, median_psnr)

    # 绘制直方图
    fig, ax1 = plt.subplots()
    ax1.hist(psnr_values, bins=bins, alpha=0.5, color='steelblue', density=True)
    ax1.axvline(x=median_psnr, color='k', label='Median', linestyle='--')
    ax1.axvline(x=mean_psnr, color='red', label='Mean', linestyle='--')
    ax1.set_xlabel('PSNR(dB)')
    ax1.set_ylabel('Probability Density')
    ax1.legend()

    # 设置x轴网格线和网格线背景色
    ax1.grid(color='white')
    ax1.xaxis.grid(True, linewidth=0.5, which='major', color='white')
    ax1.set_facecolor('lavender')
    # 平滑
    x = (bins[:-1] + bins[1:]) / 2
    f = interp1d(x, hist, kind='cubic')
    x_new = np.linspace(x[0], x[-1], num=1000)
    y_new = f(x_new)

    # 绘制直方图变化趋势折线图
    ax2 = ax1.twinx()
    ax2.plot(x_new, y_new, color='steelblue', label='Count Trend')
    ax2.set_ylim([0, max(hist) * 1.1])
    # 显示图像
    plt.show()





def ssim_show(psnr_values):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    # 生成5000个近似符合正态分布的PSNR值


    # 将PSNR值分为小区间
    # num_bins = int((1 - 0) / 0.01)
    bins = [i for i in np.arange(0.5, 1, 0.005)]

    # 统计每个区间的数量
    hist, bins = np.histogram(psnr_values, bins=bins, density=True)

    # 计算中位数和均值
    median_psnr = np.median(psnr_values)
    mean_psnr = np.mean(psnr_values)
    print(mean_psnr,median_psnr)

    # 绘制直方图
    fig, ax1 = plt.subplots()
    ax1.hist(psnr_values, bins=bins, alpha=0.5, color='steelblue', density=True)
    ax1.axvline(x=median_psnr, color='k', label='Median', linestyle='--')
    ax1.axvline(x=mean_psnr, color='red', label='Mean', linestyle='--')
    ax1.set_xlabel('SSIM')
    ax1.set_ylabel('Probability Density')
    ax1.legend()

    # 设置x轴网格线和网格线背景色
    ax1.grid(color='white')
    ax1.xaxis.grid(True, linewidth=0.5, which='major', color='white')
    ax1.set_facecolor('lavender')
    # 平滑
    x = (bins[:-1] + bins[1:]) / 2
    f = interp1d(x, hist, kind='cubic')
    x_new = np.linspace(x[0], x[-1], num=1000)
    y_new = f(x_new)

    # 绘制直方图变化趋势折线图
    ax2 = ax1.twinx()
    ax2.plot(x_new, y_new, color='steelblue', label='Count Trend')
    ax2.set_ylim([0, max(hist) * 1.1])
    # 显示图像
    plt.show()
from CSWT_Unet_train import *
class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        self.data_memmaps = [np.load(path, mmap_mode='r+') for path in data_paths]
        # self.target_memmaps = [np.load(path, mmap_mode='r+') for path in target_paths]
        self.start_indices = [0] * len(data_paths)
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]


    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        img = self.data_memmaps[memmap_index][index_in_memmap,0,:, :, :]
        label_up = self.data_memmaps[memmap_index][index_in_memmap,1,:, :, :]

        img = np.squeeze(img)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)

        label_up = np.squeeze(label_up)
        label_up = label_up.transpose(2,0,1)
        label_up = torch.from_numpy(label_up)

        return index, img,label_up

def center_crop(img, crop_size=64):
    """
    从中心裁剪图像，支持 3D (C, H, W) 或 4D (B, C, H, W)
    """
    _, h, w = img.shape if img.dim() == 3 else img.shape[1:]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[..., top:top + crop_size, left:left + crop_size] if img.dim() == 4 else img[:, top:top + crop_size,
                                                                                       left:left + crop_size]

#----------------------------------------训练的类-----------------------------------------------------#
class Trainer:
    record = {"train_loss_d": [],"train_loss_g":[],"train_psnr": [],"train_ssim": [],"val_loss": [],"val_psnr":[],"val_ssim":[]}
    x_epoch = []

    def __init__(self,args):
        self.args = args
        self.device = args.device
        self.gnet_up = CAT_Unet(
        img_size= 128,
        in_chans= 3,
        depth= [4,6,6,8],
        split_size_0= [4,4,4,4],
        split_size_1= [0,0,0,0],
        dim= 48,
        num_heads= [2,2,4,8],
        mlp_ratio= 4,
        num_refinement_blocks= 4,
        bias= False,
        dual_pixel_task= False,
        )
        batch = self.args.batch

        # self.train_loader = DataLoader(datasets.BigDataset(
        #     [self.args.train_data_path1, self.args.train_data_path2, self.args.train_data_path3,
        #      self.args.train_data_path4, self.args.train_data_path5, self.args.train_data_path6,
        #      self.args.train_data_path7]),
        #                                batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(BigDataset([self.args.val_data_path]),
                                     batch_size=1, shuffle=False, drop_last=True)
        #生成器的损失
        self.criterion_g = loss.PerceptualLoss_1(self.device)
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
        self.gnet_up.to(self.device)
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params,star training...")

            else:
                param_dict_up = torch.load(self.args.save_path)
                param_dict_down = torch.load(self.args.save_path1)
                self.epoch = 80
                self.lr = param_dict_up["lr"]
                self.gnet_up.load_state_dict(param_dict_up["gnet_dict"])


        self.real_up_label = torch.ones([batch, 1]).to(self.device)
        self.fake_down_label = torch.zeros([batch, 1]).to(self.device)
        self.fake_up_label = torch.ones([batch, 1]).to(self.device)
        self.real_down_label = torch.zeros([batch, 1]).to(self.device)
    #------------------两个静态方法计算结构相似性系数和高值信噪比---------------------------#


    @staticmethod
    def calculate_psnr(img1, img2):
        img1 = center_crop(img1)
        img2 = center_crop(img2)
        return 10 * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    @staticmethod
    def calculate_ssim(img1, img2, window_size=11, size_average=True):
        img1 = center_crop(img1)
        img2 = center_crop(img2)
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)

    #------------------------------------------------------------------------------#

    # 验证主函数
    def val(self, epoch):
        self.gnet_up.eval()
        psnrs = []
        ssims = []
        ssim = 0
        psnr = 0
        total = 0
        # files = np.load("../test_data_npy/real_blend_galaxy_files.npy")
        # data = np.zeros((580,3,128,128,3))
        print("Test start...")
        with torch.no_grad():
            d = 0
            for i, (index, img,up_label) in enumerate(self.val_loader):
                img = img.float()
                up_label = up_label.float()
                img = img.to(self.device)
                up_label = up_label.to(self.device)
                # down_label = up_label.to(self.device)
                fake_up_img = self.gnet_up(img)
                # fake_up_img = self.gnet_up(fake_up_img)
                # fake_down_img = self.gnet_up(img)
                # fake_down_img = self.gnet_down(fake_down_img)
                # fake_down_img = self.gnet_down(fake_down_img)
                ssim_x = self.calculate_ssim(fake_up_img, up_label).item()
                # ssim_y = self.calculate_ssim(fake_up_img, up_label).item()
                # ssim = (ssim_x + ssim_y) / 2
                # ssims.append(ssim)
                psnr_x = self.calculate_psnr(fake_up_img, up_label).item()


                ssim += ssim_x
                psnr += psnr_x

                total += 1
                # psnr_y = self.calculate_psnr(fake_up_img, up_label).item()
                # psnr = (psnr_y + psnr_x) / 2

                d += 1
                ssims.append(ssim_x)
                psnrs.append(psnr_x)
                # img = convert_image(img)
                # up_label = convert_image(up_label)
                # # down_label = convert_image(down_label)
                # # fake_down_img = convert_image(fake_down_img)
                # # np.save("./fake_up_img.npy",fake_up_img.reshape((1,128,128,3)))
                # # np.save("./fake_down_img.npy",fake_down_img)
                # # data[i,0,:,:,:] = img
                # # data[i,1,:,:,:] = fake_up_img
                # # data[i,2,:,:,:] = fake_down_img
                # fig = plt.figure()
                # # ssim_x = np.around(ssim_x, decimals=2)
                # # ssim_y = np.around(ssim_y, decimals=2)
                #
                # gs = GridSpec(1, 3)
                # fig = plt.figure(figsize=(8, 3))
                # # ax1 = fig.add_subplot(gs[0, 0])
                # # ax2 = fig.add_subplot(gs[1, 0])
                # ax3 = fig.add_subplot(gs[0, 0])
                # ax4 = fig.add_subplot(gs[0, 1])
                # ax5 = fig.add_subplot(gs[0, 2])
                # fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=0.9, wspace=0, hspace=0)
                # for ax in [ax3, ax4, ax5]:
                #     ax.axis('off')
                # # ax1.imshow(up_label)
                # # ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
                # # ax2.imshow(down_label)
                # # ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
                # ax3.imshow(img)
                # ax3.text(3, 10, r'Blended', color='#FFFFFF')
                # ax5.imshow(up_label)
                # ax5.text(3., 10., r'Preblend', color='#FFFFFF')
                # # ax4.text(3., 75., str(round(ssim_x,2))+"dB", color='#FFFFFF')  #
                # ax4.imshow(fake_up_img)
                # ax4.text(3., 10., r'Deblended', color='#FFFFFF')
                # ax4.text(3., 120., "ssim :" + str(round(ssim_x, 2)), color='#FFFFFF')  #
                # ax4.text(60., 120., "psnr :" + str(round(psnr_x, 2)) + " dB", color='#FFFFFF')  #
                #
                # fig.text(0.16, 0.95, "input Deblender", ha="center", va="center", fontsize=12, color='black')
                # fig.text(0.5, 0.95, "output Deblender", ha="center", va="center", fontsize=12,
                #          color='black')
                # fig.text(0.84, 0.95, "target", ha="center", va="center", fontsize=12, color='black')
                #
                # plt.tight_layout(pad=0)
                # plt.subplots_adjust(wspace=0.06, hspace=-0.42)
                # # plt.show()
                # # break
                # # plt.show()
                # fig.savefig("../test_data/0/{}.png".format(i))
                print("-------------已经完成第{}个测试 ------------------".format(i))
                    # if d >= 20:
                    #     break

                # if i>=0:
                #     break
            return np.unique(np.sort(ssims)), np.unique(np.sort(psnrs))

def main(args,k):
    t = Trainer(args)
    for epoch in range(1):
        ssims,psnrs = t.val(epoch)
        ssims = np.array(ssims)
        psnrs = np.array(psnrs)

        # np.save("./test_data/predict_data/psnr_{}.npy".format(k), psnrs)
        # np.save("./test_data/predict_data/ssim_{}.npy".format(k), ssims)
        # psnr_show(psnrs)
        # ssim_show(ssims)

if __name__ == '__main__':
    for k in range(1,7):
        parser = argparse.ArgumentParser(description="Train Galaxy Deblender Models")
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--val_data_path", default="./test_data/val_{}.npy".format(k), type=str)
        parser.add_argument("--resume", default=True, type=bool)
        parser.add_argument("--num_epochs", default=80, type=int)
        parser.add_argument("--save_path", default="./weight/decals_up_weight.pt", type=str)
        parser.add_argument("--save_path1", default="./weight/decals_up_weight.pt", type=str)
        parser.add_argument("--interval", default=20, type=int)
        parser.add_argument("--batch", default=1, type=int)
        args1 = parser.parse_args()
        main(args1,k)
