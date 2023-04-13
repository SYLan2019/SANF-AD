import math

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from subnet import *
import timm
from efficientnet_pytorch import EfficientNet

import clip
import config as c
from freia_funcs import *
from pytorch_pretrained_vit.model import ViT

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head_mlp(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': F_fully_connected,
                           'F_args': {'channels_hidden': c.fc_internal}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class ADwithGlow(nn.Module):
    def __init__(self):
        super(ADwithGlow, self).__init__()

        if c.extractor == 'VIT':
            self.feature_extractor = ViT('B_16_imagenet1k', pretrained=True)
            self.feature_extractor.fc = nn.Identity()
        elif c.extractor == 'clip':
            model, preprocess = clip.load('ViT-B/16', download_root='model/')
            self.model = model
            self.feature_extractor = model.visual
            self.feature_extractor.proj = None
        self.nf_mlp = nf_head_mlp()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

    def vit_ext(self,x):
        # B, N, C
        fem = self.feature_extractor(x)
        x_1 = fem[:, 0, :]
        x_2 = fem[:, 1:, :]
        B, N, C = x_2.shape
        H = W = int(N ** 0.5)
        # 24 x 24
        x_2 = x_2.reshape(B, C, H, W)
        #  C 8 x 8 --- C 64  -- 64 C
        x_2 = self.pool(x_2).reshape(B, C, -1).permute(0, 2, 1)
        x = torch.cat((x_1.unsqueeze(1),x_2),dim=1)
        return x

    def forward(self, x):
        if c.pretrained:
            # B,N,C ---> B,C,N
            x = x.transpose(1,2)
            z_sem = self.nf_mlp(x)
            z = z_sem.transpose(1,2)[:,0]
            return z
        else:
            raise AttributeError


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
