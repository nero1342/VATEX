# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os


from .ref_ytvos import (
    register_ref_ytvos_instances,
    _get_ref_ytvos_2021_instances_meta,
)

from .coco import register_coco_instances

# ==== Predefined splits for RefYTVIS 2021 ===========
_PREDEFINED_SPLITS_REF_YTVOS_2021 = {
    "ref_ytvos_2021_train": ("data/ref-youtube-vos/train/JPEGImages",
                         "data/train.json"),
    "ref_ytvos_2021_val_fake": ("data/ref-youtube-vos/valid/JPEGImages",
                         "data/valid_fake.json"),
    "ref_ytvos_2021_val": ("data/ref-youtube-vos/valid/JPEGImages",
                       "data/valid.json"),
    "ref_davis_val": ("data/ref-davis/valid/JPEGImages/",
                       "data/ref-davis/valid.json"),
    
    # "ref_ytvos_2021_test": ("ref_ytvos_2021/test/JPEGImages",
    #                     "ref_ytvos_2021/test.json"),
}

# ==== Predefined splits for RefCoco 2014 ===========
_PREDEFINED_SPLITS_REF_COCO_2014 = {
    "refcoco_train": ("data/coco/train2014",
                         "data/coco/refcoco/instances_refcoco_train.json"),
    "refcoco_val": ("data/coco/train2014",
                         "data/coco/refcoco/instances_refcoco_val.json"),
    "refcoco_testA": ("data/coco/train2014",
                         "data/coco/refcoco/instances_refcoco_testA.json"),
    "refcoco_testA_fake": ("data/coco/train2014",
                         "data/coco/refcoco/instances_refcoco_testA_fake.json"),
    "refcoco_testB": ("data/coco/train2014",
                         "data/coco/refcoco/instances_refcoco_testB.json"),                                          
    "refcoco+_train": ("data/coco/train2014",
                        "data/coco/refcoco+/instances_refcoco+_train.json"),
    "refcoco+_val": ("data/coco/train2014",
                        "data/coco/refcoco+/instances_refcoco+_val.json"),
    "refcoco+_testA": ("data/coco/train2014",
                        "data/coco/refcoco+/instances_refcoco+_testA.json"),
    "refcoco+_testB": ("data/coco/train2014",
                        "data/coco/refcoco+/instances_refcoco+_testB.json"),
    "refcocog_train": ("data/coco/train2014",
                         "data/coco/refcocog/instances_refcocog_train.json"),
    "refcocog_val": ("data/coco/train2014",
                         "data/coco/refcocog/instances_refcocog_val.json"),
    "refcocog_test": ("data/coco/train2014",
                         "data/coco/refcocog/instances_refcocog_test.json"),
    "refcoco_joint_train": ("data/coco/train2014",
                         "data/coco/instances_joint_train.json"),
                                       
}


def register_all_ref_ytvos_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_YTVOS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ref_ytvos_instances(
            key,
            _get_ref_ytvos_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ref_coco_2014(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_COCO_2014.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = './'
    register_all_ref_ytvos_2021(_root)
    register_all_ref_coco_2014(_root)