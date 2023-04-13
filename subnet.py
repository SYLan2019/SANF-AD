#!/usr/bin/python
# author mawei
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, 2*dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class HiLo(nn.Module):
    """
    HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim*2)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            # self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_q = nn.Linear(self.dim,self.h_dim,bias=qkv_bias)
            self.h_kv = nn.Linear(self.dim,self.h_dim*2,bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim*2)

    def hifi(self, x_1,x):
        # B,24,24,384
        B, H, W, C = x.shape
        # 24//3 = 8
        h_group, w_group = H // self.ws, W // self.ws
        # 8*8 = 64
        total_groups = h_group * w_group
        # B,8,8,3,3,384
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        #
        # qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        # b,384 --- b,192 --- b,1,4,48 --- b,4,1,48---b,1,4,1,48
        q = self.h_q(x_1).reshape(B,self.h_heads, self.h_dim // self.h_heads).unsqueeze(1).permute(0,2,1,3).unsqueeze(1)
        # -1 将中间的3x3的窗口展成9
        kv = self.h_kv(x).reshape(B, total_groups, -1, 2, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        k,v = kv[0],kv[1] #  B, hw, n_head, ws*ws, head_dim :   B, 8*8, 4, 3*3 ,48

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, total_groups, 1, self.h_dim)
        attn = torch.mean(attn,dim=1)
        # x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        # x = self.h_proj(x)
        x = self.h_proj(attn)
        return x

    def lofi(self, x_1,x):
        # 2,24,24,192
        B, H, W, C = x.shape

        q = self.l_q(x_1).unsqueeze(1).reshape(B, 1, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
        # q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x = x.permute(0, 3, 1, 2)
            # 2,24,24,192 --- 2,192,8,8 --- 2,64,192
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            # 2,64,192 --- 2,64,2,4,48 ---2,2,4,64,48
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        #2,4,64,48
        k, v = kv[0], kv[1]
        # 2,4,1,48  @ 2,4,64,48 T = 2,4,1,64
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 2,4,1,64 @ 2,4,64,48 = 2,4,1,48
        x = (attn @ v).transpose(1, 2).reshape(B, 1, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        # x : (B,N,C)  (B,577,384)
        # B, N, C = x.shape
        x_1 = x[:,0,:]
        x_2 = x[:,1:,:]
        B, N, C = x_2.shape
        H = W = int(N ** 0.5)

        x_2 = x_2.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.lofi(x_1,x_2)
            # return x.reshape(B, N, C)
            return x

        if self.l_heads == 0:
            x = self.hifi(x_1,x_2)
            # return x.reshape(B, N, C)
            return x

        hifi_out = self.hifi(x_1,x_2)
        lofi_out = self.lofi(x_1,x_2)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        # x = x.reshape(B, C, H, W)
        # x = x.reshape(B, N, C)
        return x

class HiLoBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop=0.,drop_path=0., norm_layer=nn.LayerNorm, local_ws=3, alpha=0.5):
        super(HiLoBlock, self).__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.attn = HiLo(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                         attn_drop=attn_drop,proj_drop=drop,window_size=local_ws,alpha=alpha)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        x = x.reshape(B,C,H,W)
        x = x + self.drop_path(self.attn(x))
        return x

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        # mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        mapper_x = [temp_x * (dct_h // 6) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 6) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter



if __name__ == '__main__':

    # x = torch.randn((2,577,384))
    x = torch.randn((2,400,24,24))
    net = MultiSpectralAttentionLayer(channel=400,dct_h=24,dct_w=24)
    # net = HiLo(dim=384,num_heads=8,window_size=3,alpha=1)
    # net = Class_Attention(dim=512,num_heads=4)
    # alpha 调节的是 lofi的比例，alpha =0 纯窗口注意力机制 alpha = 1
    y = net(x)
    print(y.shape)
