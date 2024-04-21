# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5, num_classes = -1):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = num_classes
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target

logger = logging.getLogger(__name__)
class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM if is_train else 1 
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        # temp = copy.deepcopy(dataset_dict)
        # try:
        anno_id = dataset_dict["anno_id"]
        exp_id = dataset_dict["exp_id"]
       
        imageType = False 
        if "length" in dataset_dict: # Video dataset
            file_names = dataset_dict["file_names"]

            video_length = dataset_dict["length"]
            video_id = dataset_dict["video_id"]
            video_name = dataset_dict["video_name"]
            video_annos = dataset_dict["annotations"]

            expressions = dataset_dict["expressions"]
            expression = dataset_dict["expression"]

        else: # Image dataset 
            imageType = True
            file_names = [dataset_dict["file_name"]]

            video_length = 1
            video_name = str(dataset_dict["id"])
            video_id = str(dataset_dict["image_id"])
            video_annos = [dataset_dict["annotations"]]

            expressions = dataset_dict["annotations"]
            expression = expressions[anno_id]["expressions"][exp_id] 

        
        _dataset_dict = {
            "height": dataset_dict["height"],
            "width": dataset_dict["width"],
            "length": video_length,
            "video_name": video_name,
            "video_id": video_id,
            "anno_id": anno_id, #dataset_dict["org_ann_id"], #anno_id,
            "exp_id": exp_id,
            "expression": expression,
        }
        
        if self.is_train or video_length < self.sampling_frame_num:
            while True:
                selected_idx = np.random.choice(
                    range(video_length), self.sampling_frame_num
                )
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)

                _ids = set()
                for frame_idx in selected_idx:
                    _ids.update([anno["id"] for anno in video_annos[frame_idx]])

                if len(_ids) == 0:
                    continue 
                break 
        else:
            selected_idx = range(video_length)

        if self.is_train:     
            # Positive 
            positive_exp = random.choice(expressions[anno_id]["expressions"])
            if positive_exp == expression:
                positive_exp = positive_exp + '.'
            # Negative
            negative_id = random.choice(list(_ids)) 
            if negative_id == anno_id:
                negative_id = -1 
                negative_exp = "None"
            else:
                negative_exp = random.choice(expressions[negative_id]["expressions"])
            ids = { anno_id: 0,  negative_id: 1}
            _dataset_dict["positive_exp"] = positive_exp
            _dataset_dict["negative_exp"] = negative_exp

        _dataset_dict["image"] = []
        _dataset_dict["instances"] = []
        _dataset_dict["file_names"] = []

        for frame_idx in selected_idx:
            _dataset_dict["file_names"].append(file_names[frame_idx])
            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            # utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # image_shape = (480, 480)  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            _dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                if anno["id"] not in ids:
                    continue 
                _anno = {k : copy.deepcopy(v) for k, v in anno.items()}
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]
            _gt_classes = [int(_anno['category_id'] <= 0) for _anno in sorted_annos]
            
            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            # print(instances.gt_masks.tensor.shape)
            instances.gt_ids = torch.tensor(_gt_ids)
            instances.gt_classes = torch.tensor(_gt_classes)

            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances, self.num_classes)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))

            _dataset_dict["instances"].append(instances)
        return _dataset_dict
