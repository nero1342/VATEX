import numpy as np
import os
from refer import REFER
import cv2  
from tqdm import tqdm
import json
import pickle
import json
from collections import defaultdict 

def convert_to_coco(data_root='data/coco', output_root='data/coco', dataset='refcoco', dataset_split='unc'):
    print(f"Convert {dataset}-{dataset_split} from {data_root} to {output_root}...")
    dataset_dir = os.path.join(data_root, dataset)
    output_dir = os.path.join(output_root, dataset) # .json save path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read REFER
    refer = REFER(data_root, dataset, dataset_split)
    refs = refer.Refs
    anns = refer.Anns
    imgs = refer.Imgs
    cats = refer.Cats
    sents = refer.Sents
    """
    # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
    Refs: List[Dict], "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
                      "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
    Anns: List[Dict], "segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"
    Imgs: List[Dict], "license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"
    Cats: List[Dict], "supercategory", "name", "id"
    Sents: List[Dict], "tokens"(List), "raw", "sent_id", "sent", here the "sent_id" is consistent
    """
    print('Dataset [%s_%s] contains: ' % (dataset, dataset_split))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('There are %s expressions for %s refereed objects in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

    print('\nAmong them:')
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
    elif dataset == 'refcoco+':
        splits = ['train', 'val',  'testA', 'testB']
    elif dataset == 'refcocog':
        splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.

    for split in splits:
        ref_ids = refer.getRefIds(split=split)
        print('     %s referred objects are in split [%s].' % (len(ref_ids), split))

    with open(os.path.join(dataset_dir, "instances.json"), "r") as f:
        ann_json = json.load(f)

    # 1. for each split: train, val...
    for split in splits:
        max_length = 0 # max length of a sentence

        coco_ann = {
            "info": "",
            "licenses": "",
            "images": [],   # each caption is a image sample
            "annotations": [],
            "categories": []
        }
        coco_ann['info'], coco_ann['licenses'], coco_ann['categories'] = \
                                    ann_json['info'], ann_json['licenses'], ann_json['categories']
        ref_ids = refer.getRefIds(split=split)

        imgs_id = dict()
        anns_id = dict()
        max_length = 0 
        for i in tqdm(ref_ids): 
            ref = refs[i]
            # "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
            #             "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
            img = imgs[ref["image_id"]]
            ann = anns[ref["ann_id"]]

            if ref["image_id"] not in imgs_id:
                image_info = {
                    "file_name": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                    "original_id": img["id"],
                    "id": len(imgs_id),
                    "dataset_name": dataset
                }
                imgs_id[img["id"]] = len(imgs_id)
                coco_ann["images"].append(image_info)

            if ref["ann_id"] not in anns_id:
                ann_info = {
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "bbox": ann["bbox"],
                    "image_id": imgs_id[img["id"]],
                    "category_id": ann["category_id"],
                    "id": len(anns_id),
                    "original_id": ann["id"],
                    "expressions": []
                }
                anns_id[ann["id"]] = len(anns_id)
                coco_ann["annotations"].append(ann_info)

            for sentence in ref["sentences"]: 
                exp = sentence["sent"] 
                coco_ann["annotations"][anns_id[ann["id"]]]["expressions"].append(exp)
                max_length = max(max_length, len(sentence["tokens"]))
                
        print("Max sentence length of the split: ", max_length)
        print(len(coco_ann["images"]), len(coco_ann["annotations"]), sum([len(x["expressions"]) for x in coco_ann["annotations"]]))
        # save the json file
        save_file = "instances_{}_{}.json".format(dataset, split)
        with open(os.path.join(output_dir, save_file), 'w') as f:
            json.dump(coco_ann, f)

if __name__ == '__main__':
    data_root = 'data/coco'
    datasets = ["refcoco", "refcoco+", "refcocog"]
    datasets_split = ["unc", "unc", "umd"]
    for (dataset, dataset_split) in zip(datasets, datasets_split):
        convert_to_coco(data_root = data_root, output_root = data_root, dataset=dataset, dataset_split=dataset_split)
        print("")
