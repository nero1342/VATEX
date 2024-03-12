#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained swin model:
  wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl
  # Then, use swin_tiny_patch4_window7_224.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224.pkl"
INPUT:
  FORMAT: "RGB"
"""
from pprint import pprint 
if __name__ == "__main__":
    input = sys.argv[1]

    state_dict = torch.load(input, map_location="cpu")["state_dict"]


    # extract swinT's kinetics-400 pretrained weights and ignore the rest (prediction head etc.)
    # sum over the patch embedding weight temporal dim  [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
    patch_embed_weight = state_dict['backbone.patch_embed.proj.weight']
    patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
    state_dict['backbone.patch_embed.proj.weight'] = patch_embed_weight

    mapping = {}
    keys = list(state_dict.keys())
    for k in keys:
        if "downsample." in k:
          newK = k.replace("downsample.", "").replace("layers", "downsamples")
          state_dict[newK] = state_dict.pop(k)

    # print(patch_embed_weight.shape)


    pprint(state_dict.keys())
    res = {"model": state_dict, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)