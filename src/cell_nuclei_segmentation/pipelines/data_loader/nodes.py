import pandas as pd
from kedro.io import PartitionedDataSet
import numpy as np
import cv2
import mmengine
import mmcv

def annotate_dataset(images : PartitionedDataSet, gt_images : PartitionedDataSet):
    
    annotations = []
    all_images = []
    obj_count = 0

    for i, image_name in enumerate(images):
        image = np.array(images[image_name]())
        gt_image = np.array(gt_images[image_name]())
        gt_countours = get_countours(gt_image)

        height, width = image.shape[:2]
        all_images.append(dict(
            id=i,
            file_name=image_name + '.tif',
            height=height,
            width=width))
        
        for region in gt_countours:

            px = list([int(x) for x in region[:,0,0]])
            py = list([int(y) for y in region[:,0,1]])
            if len(px) == 4 or len(px) == 2:
                continue
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=i,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=all_images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'nuclei'}])

    return coco_format_json


def convert_cells_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmengine.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmengine.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'balloon'}])
    mmengine.dump(coco_format_json, out_file)

def get_countours(image):
    image_uint8 = image.astype(np.uint8)

    to_extract = np.unique(image)
    all_contours = []

    for i in range(1, len(to_extract)):
        _, thresh = cv2.threshold((image_uint8 == i).astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) for cnt in contours]
        all_contours.extend(contours)

    return all_contours