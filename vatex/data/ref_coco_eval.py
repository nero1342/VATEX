# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
from re import S
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from .datasets.ref_ytvos_api.ytvos import YTVOS
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from PIL import Image,ImageDraw

class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        
        with contextlib.redirect_stdout(io.StringIO()):
            # self._ytvis_api = YTVOS(json_file)
            from pycocotools.coco import COCO
            self._ytvis_api = COCO(json_file)
        

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs)

        for i, p in enumerate(prediction):
            pred = p["segmentations"]
            # pred = pred[[0]]
            # gt = self._ytvis_api.loadAnnsFromExps(p["id"])[0]
            try:
                ann = self._ytvis_api.imgToAnns[int(p["video_id"])][p["anno_id"]]
                gt = self._ytvis_api.annToMask(ann)[None,]
                # print(gt.shape, pred.shape)
                gt = cv2.resize(gt[0], (pred.shape[2], pred.shape[1]), interpolation=cv2.INTER_NEAREST)[None,]
    
                # p["gt"] = gt
                gt = F.interpolate(torch.from_numpy(gt)[None,], pred.shape[1:], mode='nearest')[0].numpy()
                # p["gt"] = gt
                p["iou"], p["inter"], p["union"] = db_eval_iou(pred, gt)
    
                # p["boundary"] = db_eval_boundary(pred, gt)
                # p["segmentations"] = gt 
            except Exception as err:
                #print(f"Unexpected {err=}, {type(err)=}")
                #raise

                p["iou"] = 0
                p["boundary"] = 0
                p["inter"] = 0
                p["union"] = 0

            self.save_single_output(self._output_dir, p, i, True)
            del p["segmentations"]
            # del p["gt"]

        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "Annotations")
            self._logger.info("Saving results to {}".format(file_path))
            # self.save_output(file_path, predictions)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        
        self._logger.info("Evaluating...")
            
        iou = [] 
        boundary = [] 
        overall_inter = []
        overall_union = [] 
        query_count = {} 
        
        # fo = open("log.txt","w")
        from collections import defaultdict
        consistency = defaultdict(list)

        for prediction in predictions:
            iou.extend(prediction["iou"])
            # boundary.extend(prediction["boundary"]) 
            overall_inter.extend(prediction["inter"])
            overall_union.extend(prediction["union"])
            x = prediction["query_id"].item()
            if x not in query_count: 
                query_count[x] = 0
            query_count[x] += 1
         
        self._results["mean IoU"] = np.mean(np.array(iou))
        # self._results["f1"] = np.mean(np.array(boundary))
        # self._results["overall"] = (self._results["mean IoU"] + self._results["f1"]) / 2
        self._results["overall IoU"] = np.sum(np.array(overall_inter)) / np.sum(np.array(overall_union))
        self._results["overall_inter"] = np.sum(np.array(overall_inter))
        self._results["overall_union"] = np.sum(np.array(overall_union))

            
        # Precision@X
        for thres in range(5, 10):
            self._results[f"Pr@{thres * 10}"] = np.mean(np.array(iou) > thres / 10)
            
        for x, y in query_count.items():
            self._results[f"Query distribution {x}"] = y
            
    def save_single_output(self, root, prediction, obj_id, visualize = True):
        # os.makedirs(root, exist_ok=True)
        from PIL import Image 
        #print(prediction)
        # self._logger.info("Annotations are not available for evaluation.")
        video_id = str(prediction["video_id"])
        exp_id = str(prediction["exp_id"])
        ann_id = str(prediction["anno_id"])
        file_names = prediction["file_names"]
        masks = prediction["segmentations"]
        # gts = prediction["gt"]
        exp = prediction["expression"].replace("/", " ")
        score = prediction["score"]
        # dir = os.path.join(root, video_id, f"{exp_id}_{exp}_{iou}_{f1}")
        dir = os.path.join(root, video_id, exp_id)
        # dir_visualize = os.path.join(root.replace('inference', 'visualize'), video_id, exp_id + "_" + exp)
        dir_visualize = dir.replace('inference', 'visualize')
        # color_list = color_list.astype('uint8').tolist()
        for i, file_name in enumerate(file_names):
            x = file_name.split('/')[-1].split('.')[0]
            y = file_name.split('/')[-1].replace('.jpg', '.png')
            # dir = os.path.join(root, x)
            os.makedirs(dir, exist_ok=True)
            mask = Image.fromarray(masks[i] * 255).convert("P")
            # iou = prediction["iou"][i] * 100
            # mask.save(os.path.join(dir, f"{exp_id}_{iou:02f}_{file_name.split('/')[-1]}").replace('.jpg', '.png'))
            mask.save(os.path.join(dir, file_name.split('/')[-1]).replace('.jpg', '.png'))
            if visualize:
                try:
                    iou = prediction["iou"][i]
                    f1 = prediction["boundary"][i]
                except:
                    iou = f1 = 0 
                os.makedirs(dir_visualize, exist_ok=True)
                source_img = Image.open(file_name).convert("RGB")
                # source_img = Image.fromarray(overlay_davis(source_img, gts[i],colors=[0,0,0, 0, 255, 0]))
                source_img = Image.fromarray(overlay_davis(source_img, masks[i],colors=[0,0,0, 0, 255, 0]))#.convert('RGBA')
                # source_img = vis_add_mask(source_img, mask, color_list[i%len(color_list)])
                draw = ImageDraw.Draw(source_img)
                draw.text((0, 0), exp, (255,255,255))
                source_img.save(os.path.join(dir_visualize, file_name.split('/')[-1]).replace('.jpg', '.png'))
                # source_img.save(os.path.join(dir_visualize, f"{iou:02f}_{score:02f}_{exp}_{video_id}_{exp_id}_{y}"))

def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_name"]
    video_length = inputs[0]["length"]
    exp = inputs[0]["expression"]
    file_names = inputs[0]["file_names"]
    exp_id = inputs[0]["exp_id"]
    id = inputs[0]["video_id"]
    anno_id = inputs[0]["anno_id"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    qid = outputs["pred_ids"]

    ytvis_results = []
    for instance_id, (s, l, m, q) in enumerate(zip(scores, labels, masks, qid)):
        m = np.array(m).astype(np.uint8)
        res = {
            "id": id, 
            "file_names": file_names,
            "expression": exp,
            "video_id": video_id,
            "anno_id": anno_id,
            "exp_id": exp_id,
            "score": s,
            "category_id": l,
            "segmentations": m,
            "query_id": q
        }
        ytvis_results.append(res)

    return ytvis_results


import math
import numpy as np
import cv2

import torch.nn.functional as F

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    Return:
        jaccard (float): region similarity
    """
    
    # print(annotation.shape, segmentation.shape)
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    def _computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U
    inters, union = _computeIoU(annotation, segmentation)
    # Intersection between all sets
    # inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    # union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    if union == 0:
        j = 0
    else: 
        j = inters * 1.0 / union

    return [j], [inters], [union]


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels
    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.6):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    image = np.array(image)

    colors = np.reshape(colors, (-1, 3))
    # colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)

