# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from einops import rearrange, repeat

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine2D
from ..attention import BAT, _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn

from .msdeformattn import MSDeformAttnTransformerEncoderOnly


@SEM_SEG_HEADS_REGISTRY.register()
class ContextualMultimodalDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine2D(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = 3 # int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                conv_dim if idx > 0 else in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        fusion_fpn_modules = []
        fusion_modules = []

        for idx in range(len(self.transformer_in_features)):
            fusion_module = BAT(conv_dim, nhead=8)

            for p in fusion_module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            
            stage = int(idx + 1)
            self.add_module("fusion_{}".format(stage), fusion_module)
            fusion_modules.append(fusion_module)
        
        fusion_modules2 = []

        for idx in range(len(self.transformer_in_features) + 1):
            fusion_module = BAT(conv_dim, nhead=8)

            for p in fusion_module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            
            stage = int(idx + 1)
            self.add_module("fusion2_{}".format(stage), fusion_module)
            fusion_modules2.append(fusion_module)
        
        self.fusion_modules2 = fusion_modules2
        self.fusion_modules = fusion_modules
        self.fusion_fpn_modules = fusion_fpn_modules


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features, caption):
        # for x in features:
        #     print(features[x].shape)
        text_word_features, text_word_masks, text_pos, _ = caption
        srcs = []
        pos = []

        b = text_word_features.shape[1]
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            t = x.shape[0] // b
            
            src_proj_l = self.input_proj[idx](x)    
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            src_proj_l,  _ = self.fusion_modules[idx](tgt=src_proj_l,
                                             memory=text_word_features.float(),
                                             memory_key_padding_mask=text_word_masks.float(),
                                             pos=text_pos.float(),
                                             query_pos=None
            ) 
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)# .float()

            srcs.append(src_proj_l)
            pos.append(self.pe_layer(x))

        
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        features["res3"] = out[-1]
        features["res4"] = out[-2]

        texts = [text_word_features.sum(0) / (1 - text_word_masks.float()).sum()]
        c, h, w = out[-3].shape[-3:]
        src_proj_l = rearrange(out[-3], '(b t) c h w -> (t h w) b c', b=b, t=t)
        src_proj_l,  text_word_features = self.fusion_modules2[-1](tgt=src_proj_l,
                                            memory=text_word_features.float(),
                                            memory_key_padding_mask=text_word_masks.float(),
                                            pos=text_pos.float(),
                                            query_pos=None
        ) 
        src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)# .float()
        out = [src_proj_l]



        texts.append(text_word_features.sum(0) / (1 - text_word_masks.float()).sum())
        # out.append(src_proj_l)
        # out = [out[-3]]
        # out = out[:-2]
        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)

        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            # print(idx, f, x.shape)
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            y = lateral_conv(x)

            # # Following FPN implementation, we use nearest upsampling here
            
            c, h, w = y.shape[-3:]
            y = rearrange(y, '(b t) c h w -> (t h w) b c', b=b, t=t)
            y,  text_word_features = self.fusion_modules2[idx](tgt=y,
                                             memory=text_word_features.float(),
                                             memory_key_padding_mask=text_word_masks.float(),
                                             pos=text_pos.float(),
                                             query_pos=None


            ) 
            y = rearrange(y, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)# .float()

            y = y + F.interpolate(out[-1], size=y.shape[-2:], mode="bilinear", align_corners=False)

            y = output_conv(y)
            out.append(y)


            texts.append(text_word_features.sum(0) / (1 - text_word_masks.float()).sum())
        # exit(0)
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        
        caption = (text_word_features, text_word_masks, text_pos, text_word_features.sum(0) / (1 - text_word_masks.float()).sum())

        return self.mask_features(out[-1]), out[0], multi_scale_features, caption, texts
