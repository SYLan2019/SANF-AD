"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer, AnomalyTransformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            positional_embedding: str = '1d',
            in_channels: int = 3,
            image_size: Optional[int] = None,
            num_classes: Optional[int] = None,
            add_rotation_token: bool = False,
            resize_positional_embedding: bool =False,
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        self.add_rotation_token = add_rotation_token
        if self.add_rotation_token:
            self.rotation_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            if not self.add_rotation_token:
                seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
                strict=False
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding,
                        std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x, output_layer_ind=-1):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d

        if self.add_rotation_token:
            x = torch.cat((self.rotation_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        else:
            if hasattr(self, 'class_token'):
                x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d

        x = self.transformer(x, output_layer_ind=output_layer_ind)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            # 只取了第一个patch的特征
            x1 = self.norm(x)[:, 0]
            x2 = self.norm(x)[:,1:]
            # b,d - output of rotation head or classification head
            # x = self.fc(x1)# b,num_classes
            # b, spa, c = x2.shape
            # h = int(math.sqrt(spa))
            # x2 = x2.transpose(1, 2).reshape(b, c, h, h)#空间就换成x2
            # x = x.reshape(b,768,24,24)
        return self.norm(x)


class AnomalyViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """
    # name = 'B_16_imagenet1k'，pretrained =True
    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            positional_embedding: str = '1d',
            in_channels: int = 3,
            image_size: Optional[int] = None,
            num_classes: Optional[int] = None,
            add_rotation_token: bool = False,
            clone_block_ind=-1
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            # (16,16)
            patches = config['patches']
            # 768
            dim = config['dim']
            # 3072
            ff_dim = config['ff_dim']
            # 12
            num_heads = config['num_heads']
            # 12
            num_layers = config['num_layers']
            # 0.0
            attention_dropout_rate = config['attention_dropout_rate']
            # 0.1
            dropout_rate = config['dropout_rate']
            # 768
            representation_size = config['representation_size']
            # token
            classifier = config['classifier']
            if image_size is None:
                # 384
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                # 1000
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        # Image and patch sizes
        # 384,384
        h, w = as_tuple(image_size)  # image sizes
        # 16,16
        fh, fw = as_tuple(patches)  # patch sizes
        # 24
        gh, gw = h // fh, w // fw  # number of patches
        # 24*24
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        self.add_rotation_token = add_rotation_token
        if self.add_rotation_token:
            self.rotation_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            if not self.add_rotation_token:
                seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            # x + (1,seq_len,dim)(x + (1,24*24,768))
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        # -1
        self.clone_block_ind = clone_block_ind
        # 12,768,12,3072,0.1
        # 具有12个block的vit模型，相邻block的模型中大小是一样的，未作任何改变
        self.transformer = AnomalyTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                              ff_dim=ff_dim, dropout=dropout_rate)

        # self.transformer = AnomalyTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
        #                                       ff_dim=ff_dim, dropout=dropout_rate,
        #                                       clone_block_ind = self.clone_block_ind)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            # 768
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        # 768,1000
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            # 1000
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            # 384
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            # name = 'B_16_imagenet1k'
            # 往self加载预训练权重
            load_pretrained_weights(
                self, name,
                # True
                load_first_conv=(in_channels == pretrained_num_channels),
                # True
                load_fc=(num_classes == pretrained_num_classes),
                # False
                load_repr_layer=load_repr_layer,
                # False
                resize_positional_embedding=(image_size != pretrained_image_size),
                strict=False
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding,
                        std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x, output_layer_ind=-1):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d

        if self.add_rotation_token:
            x = torch.cat((self.rotation_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        else:
            if hasattr(self, 'class_token'):
                x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d

        # origin_block_outputs, cloned_block_outputs = self.transformer(x, output_layer_ind = output_layer_ind)  # b,gh*gw+1,d
        origin_block_outputs, cloned_block_outputs = self.transformer(x)  # b,gh*gw+1,d
        # if hasattr(self, 'pre_logits'):
        #     x = self.pre_logits(x)
        #     x = torch.tanh(x)
        # if hasattr(self, 'fc'):
        #     x = self.norm(x)[:, 0]  # b,d - output of rotation head or classification head
        #     x = self.fc(x)  # b,num_classes
        return origin_block_outputs, cloned_block_outputs
