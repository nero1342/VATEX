import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from typing import Tuple, Union

from detectron2.layers.blocks import FrozenBatchNorm2d
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

import clip 


@BACKBONE_REGISTRY.register()
class CLIPResnet(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()  
        self._out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
        depth = {
            "RN50": 50,
            "RN101": 101 
        }[cfg.MODEL.BACKBONE.VERSION]
        # depth = cfg.MODEL.BACKBONE.DEPTH
        self.depth = depth 

        num_blocks_per_stage = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }[depth]
        
        layers = num_blocks_per_stage
        width = 64 
        
        # default configs of CLIP ModifiedResNet, but not used if only building ModifiedResNet as backbone
        embed_dim = {
            50: 1024,
            101: 512,
        }[depth] 
        heads = width * 32 // 64
        image_resolution = 224 
        self.resnet = ModifiedResNet(
            layers=layers,
            output_dim=embed_dim,
            heads=heads,
            width=width
        )

        FrozenBatchNorm2d.convert_frozen_batchnorm(self.resnet)

        self.num_layers = len(num_blocks_per_stage)
        num_features = [int(256 * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        embed_dim = 256 
        self._out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
        # assert (
        #     "RN" in cfg.MODEL.BACKBONE.VERSION
        #     ), "Supports only ResNet version"
        # clip_model, _ = clip.load(cfg.MODEL.BACKBONE.VERSION)
        # self.clip_visual = clip_model.visual
        # self.clip_visual.attnpool = nn.Identity() 

        # FrozenBatchNorm2d.convert_frozen_batchnorm(self.clip_visual)

        num_features = [int(embed_dim * 2 ** i) for i in range(4)]
        self.num_features = num_features

        self._out_feature_strides = {
            "r2": 4,
            "r3": 8,
            "r4": 16,
            "r5": 32,
        }
        
        self._out_feature_channels = {
            "r2": self.num_features[0],
            "r3": self.num_features[1],
            "r4": self.num_features[2],
            "r5": self.num_features[3],
        }

        self.load_weights(cfg.MODEL.BACKBONE.VERSION) 

    def load_weights(self, weight_path):
        clip_model, _ = clip.load(weight_path)
        # print(
        self.resnet.load_state_dict(clip_model.visual.state_dict(),strict=False)
        # )
        # load weight here
        

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"Resnet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}

        y = self.resnet(x) 

        # def stem(x):
        #     x = self.clip_visual.relu1(self.clip_visual.bn1(self.clip_visual.conv1(x)))
        #     x = self.clip_visual.relu2(self.clip_visual.bn2(self.clip_visual.conv2(x)))
        #     x = self.clip_visual.relu3(self.clip_visual.bn3(self.clip_visual.conv3(x)))
        #     x = self.clip_visual.avgpool(x)
        #     return x

        # x = x.type(self.clip_visual.conv1.weight.dtype)
        # x = stem(x)
        # res2 = self.clip_visual.layer1(x)
        # res3 = self.clip_visual.layer2(res2)
        # res4 = self.clip_visual.layer3(res3)
        # res5 = self.clip_visual.layer4(res4)
        # x = self.clip_visual.attnpool(res5) 

        # y = {
        #     "r2": res2,
        #     "r3": res3,
        #     "r4": res4,
        #     "r5": res5,
        #     # "global": x 
        # }  

        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", norm_layer(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = norm_layer(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # embed_dim = width * 32  # the ResNet feature dimension
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        layers = [Bottleneck(self._inplanes, planes, stride, norm_layer)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, 1, norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        # x = self.attnpool(x)

        # return x
        return {
            "r2": res2,
            "r3": res3,
            "r4": res4,
            "r5": res5,
            # "global": x 
        }  
