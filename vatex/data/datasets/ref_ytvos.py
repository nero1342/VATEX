# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ref_ytvos_json", "register_ref_ytvos_instances"]


REF_YTVOS_CATEGORIES_2021 =[
  {'color': [106, 0, 228], 'id': 1, 'isthing': 1, 'name': 'penguin'},
 {'color': [174, 57, 255], 'id': 2, 'isthing': 1, 'name': 'bus'},
 {'color': [255, 109, 65], 'id': 3, 'isthing': 1, 'name': 'sedan'},
 {'color': [0, 0, 192], 'id': 4, 'isthing': 1, 'name': 'giant_panda'},
 {'color': [0, 0, 142], 'id': 5, 'isthing': 1, 'name': 'ape'},
 {'color': [255, 77, 255], 'id': 6, 'isthing': 1, 'name': 'zebra'},
 {'color': [120, 166, 157], 'id': 7, 'isthing': 1, 'name': 'shark'},
 {'color': [209, 0, 151], 'id': 8, 'isthing': 1, 'name': 'airplane'},
 {'color': [0, 226, 252], 'id': 9, 'isthing': 1, 'name': 'leopard'},
 {'color': [179, 0, 194], 'id': 10, 'isthing': 1, 'name': 'person'},
 {'color': [174, 255, 243], 'id': 11, 'isthing': 1, 'name': 'motorbike'},
 {'color': [110, 76, 0], 'id': 12, 'isthing': 1, 'name': 'bird'},
 {'color': [73, 77, 174], 'id': 13, 'isthing': 1, 'name': 'frisbee'},
 {'color': [250, 170, 30], 'id': 14, 'isthing': 1, 'name': 'whale'},
 {'color': [0, 125, 92], 'id': 15, 'isthing': 1, 'name': 'snowboard'},
 {'color': [107, 142, 35], 'id': 16, 'isthing': 1, 'name': 'plant'},
 {'color': [0, 82, 0], 'id': 17, 'isthing': 1, 'name': 'cow'},
 {'color': [72, 0, 118], 'id': 18, 'isthing': 1, 'name': 'giraffe'},
 {'color': [182, 182, 255], 'id': 19, 'isthing': 1, 'name': 'lizard'},
 {'color': [255, 179, 240], 'id': 20, 'isthing': 1, 'name': 'fox'},
 {'color': [119, 11, 32], 'id': 21, 'isthing': 1, 'name': 'train'},
 {'color': [0, 60, 100], 'id': 22, 'isthing': 1, 'name': 'frog'},
 {'color': [0, 0, 230], 'id': 23, 'isthing': 1, 'name': 'skateboard'},
 {'color': [130, 114, 135], 'id': 24, 'isthing': 1, 'name': 'elephant'},
 {'color': [165, 42, 42], 'id': 25, 'isthing': 1, 'name': 'fish'},
 {'color': [220, 20, 60], 'id': 26, 'isthing': 1, 'name': 'paddle'},
 {'color': [100, 170, 30], 'id': 27, 'isthing': 1, 'name': 'boat'},
 {'color': [183, 130, 88], 'id': 28, 'isthing': 1, 'name': 'hedgehog'},
 {'color': [134, 134, 103], 'id': 29, 'isthing': 1, 'name': 'truck'},
 {'color': [5, 121, 0], 'id': 30, 'isthing': 1, 'name': 'snail'},
 {'color': [133, 129, 255], 'id': 31, 'isthing': 1, 'name': 'tiger'},
 {'color': [188, 208, 182], 'id': 32, 'isthing': 1, 'name': 'others'},
 {'color': [145, 148, 174], 'id': 33, 'isthing': 1, 'name': 'parrot'},
 {'color': [255, 208, 186], 'id': 34, 'isthing': 1, 'name': 'cat'},
 {'color': [166, 196, 102], 'id': 35, 'isthing': 1, 'name': 'monkey'},
 {'color': [0, 80, 100], 'id': 36, 'isthing': 1, 'name': 'owl'},
 {'color': [0, 0, 70], 'id': 37, 'isthing': 1, 'name': 'toilet'},
 {'color': [0, 143, 149], 'id': 38, 'isthing': 1, 'name': 'mouse'},
 {'color': [0, 228, 0], 'id': 39, 'isthing': 1, 'name': 'sign'},
 {'color': [199, 100, 0], 'id': 40, 'isthing': 1, 'name': 'umbrella'},
 {'color': [106, 0, 228], 'id': 41, 'isthing': 1, 'name': 'tennis_racket'},
 {'color': [174, 57, 255], 'id': 42, 'isthing': 1, 'name': 'snake'},
 {'color': [255, 109, 65], 'id': 43, 'isthing': 1, 'name': 'turtle'},
 {'color': [0, 0, 192], 'id': 44, 'isthing': 1, 'name': 'duck'},
 {'color': [0, 0, 142], 'id': 45, 'isthing': 1, 'name': 'eagle'},
 {'color': [255, 77, 255], 'id': 46, 'isthing': 1, 'name': 'sheep'},
 {'color': [120, 166, 157], 'id': 47, 'isthing': 1, 'name': 'hand'},
 {'color': [209, 0, 151], 'id': 48, 'isthing': 1, 'name': 'hat'},
 {'color': [0, 226, 252], 'id': 49, 'isthing': 1, 'name': 'rabbit'},
 {'color': [179, 0, 194], 'id': 50, 'isthing': 1, 'name': 'camel'},
 {'color': [174, 255, 243], 'id': 51, 'isthing': 1, 'name': 'deer'},
 {'color': [110, 76, 0], 'id': 52, 'isthing': 1, 'name': 'crocodile'},
 {'color': [73, 77, 174], 'id': 53, 'isthing': 1, 'name': 'dog'},
 {'color': [250, 170, 30], 'id': 54, 'isthing': 1, 'name': 'parachute'},
 {'color': [0, 125, 92], 'id': 55, 'isthing': 1, 'name': 'knife'},
 {'color': [107, 142, 35], 'id': 56, 'isthing': 1, 'name': 'dolphin'},
 {'color': [0, 82, 0], 'id': 57, 'isthing': 1, 'name': 'raccoon'},
 {'color': [72, 0, 118], 'id': 58, 'isthing': 1, 'name': 'surfboard'},
 {'color': [182, 182, 255], 'id': 59, 'isthing': 1, 'name': 'lion'},
 {'color': [255, 179, 240], 'id': 60, 'isthing': 1, 'name': 'earless_seal'},
 {'color': [119, 11, 32], 'id': 61, 'isthing': 1, 'name': 'squirrel'},
 {'color': [0, 60, 100], 'id': 62, 'isthing': 1, 'name': 'horse'},
 {'color': [0, 0, 230], 'id': 63, 'isthing': 1, 'name': 'bike'},
 {'color': [130, 114, 135], 'id': 64, 'isthing': 1, 'name': 'bear'},
 {'color': [165, 42, 42], 'id': 65, 'isthing': 1, 'name': 'bucket'}]


def _get_ref_ytvos_2021_instances_meta():
    thing_ids = [k["id"] for k in REF_YTVOS_CATEGORIES_2021 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in REF_YTVOS_CATEGORIES_2021 if k["isthing"] == 1]
    assert len(thing_ids) == 65, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 65]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in REF_YTVOS_CATEGORIES_2021 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret
    
def load_ref_ytvos_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from .ref_ytvos_api.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None

    vid_ids = sorted(ytvis_api.vids.keys())

    vids = ytvis_api.loadVids(vid_ids)

    anns = {vid_id : ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids}

    total_num_valid_anns = sum([len(x) for x in anns.values()])
 
    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))
    logger.info("Loaded {} annotations in YTVIS format from {}".format(total_num_valid_anns, json_file))
    logger.info("Loaded {} expressions in YTVIS format from {}".format(len(ytvis_api.exps), json_file))
    
    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    annos = ytvis_api.anns
    vid_annos = {}
    
    expressions = ytvis_api.exps

    for id in expressions:
        expression = expressions[id]

        # print(expression)

        record = {} 
        
        v_id = expression["vid_id"]
        vid_dict = ytvis_api.loadVids([v_id])[0]

        record = {}
        file_names = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["id"] = id 
        record["file_names"] = [f for f in file_names]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        ann = anns[v_id]
        # ann = {x['id'] : x for x in ann}
        record["expressions"] = {
            x['id']: {"expressions" : x['expressions']} for x in ann
        }

        anno_id = expression.get("anno_id", None)
        record["anno_id"] = anno_id

        video_objs = []
        if anno_id is not None and 'train' in dataset_name:
            # Select frame with at least one instance
            record["file_names"] = [] 
            # anno = ytvis_api.loadAnns([anno_id])[0]
            
            for frame_idx in range(record["length"]):
                record["file_names"].append(file_names[frame_idx])
                frame_objs = []
                for anno in ann: 
                    assert anno["video_id"] == video_id

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    _bboxes = anno.get("bboxes", None)
                    _segm = anno.get("segmentations", None)
                    
                    if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                        continue
                    
                
                    bbox = _bboxes[frame_idx]
                    segm = _segm[frame_idx]

                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    elif segm:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                    if id_map:
                        obj["category_id"] = id_map[obj["category_id"]]

                    frame_objs.append(obj)
                video_objs.append(frame_objs)

        
        record["annotations"] = video_objs


        record["length"] = len(record["file_names"])
        record["expression"] = expression["expression"]
        record["exp_id"] = int(expression["exp_id"])
        record["video_name"] = vid_dict["name"]
        dataset_dicts.append(record)


    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_ref_ytvos_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ref_ytvos_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )


if __name__ == "__main__":

    cate = {} 
    for x in REF_YTVOS_CATEGORIES_2021:
        cate[x['name']] = x['id']
    
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ref_ytvos_2021_train")
    json_file = "/home/nero/YoutubeVOS/RVOS/datasets/ref_ytvos_2021/train.json"
    image_root = "/home/nero/YoutubeVOS/RVOS/datasets/ref_ytvos_2021/train/JPEGImages"
    dicts = load_ref_ytvos_json(json_file, image_root, dataset_name="ref_ytvos_2021_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]
        return frame_dic

    print(cate)
    cnt = 0
    from tqdm import tqdm
    for d in tqdm(dicts):
        vid_name = d["file_names"][0].split('/')[-2]
        # print(d['file_names'][0], len(d['annotations']), len(d["file_names"]))
        from pprint import pprint
        if len(d['annotations']) != len(d["file_names"]):
            pprint(d['annotations'])
            assert 0 == 1
        from detectron2.data import detection_utils as utils
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            # image = utils.read_image(file_name, format="RGB")
            # # print(dataset_dict["file_names"], image.shape, dataset_dict['height'], dataset_dict['width'])
            # try:
            #     utils.check_image_size(d, image)
            # except:
            #     print(file_name)
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
        cnt = cnt + 1
        if cnt == 10:
            break
