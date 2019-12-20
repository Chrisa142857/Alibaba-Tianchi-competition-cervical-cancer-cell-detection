import itertools

import json 
import os
import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

def table_classwise(cocoEval, coco):
    precisions = cocoEval.eval['precision']
    catIds = coco.getCatIds()
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(catIds) == precisions.shape[2]

    results_per_category = []
    for idx, catId in enumerate(catIds):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = coco.loadCats(catId)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float('nan')
        results_per_category.append(
            ('{}'.format(nm['name']),
                '{:0.3f}'.format(float(ap * 100))))

    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    headers = ['category', 'AP'] * (N_COLS // 2)
    results_2d = itertools.zip_longest(
        *[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print(table.table)

def cls11map6(gt_11cls):
    gt_cls11map6 = {}
    gt_cls11map6['info'] = gt_11cls['info']
    gt_cls11map6['license'] = gt_11cls['info']
    gt_cls11map6['images'] = gt_11cls['images']
    gt_cls11map6['annotations'] = []
    gt_cls11map6['categories'] = [{'id': cat6cls_id[key], 'name': key} for key in cat6cls_id]
    for ann in gt_11cls['annotations']:
        gt_cls11map6['annotations'].append({
            "id": ann['id'],
            "image_id": ann['image_id'],
            "category_id": cat_id11map6[ann['category_id']-1],
            "segmentation": ann['segmentation'],
            "bbox": ann['bbox'],
            "iscrowd": ann['iscrowd'],
            "area": ann['area']
        })
    return gt_cls11map6

cat6cls_id = {'ASC-H':1,'ASC-US':2,'HSIL':3,'LSIL':4,'Candida':5,'Trichomonas':6}
cat11cls_id = {'ASC-H1':1,'ASC-H3':2,'ASC-US1':3,'ASC-US3':4,'HSIL1':5,'HSIL3':6,'LSIL1':7,'LSIL3':8,'Candida1':9,'Candida3':10,'Trichomonas1':11}
cat_id11map6 = [1,1,2,2,3,3,4,4,5,5,6]

res_fn = '/home/admin/jupyter/wei/mmdetection-master/f_r50_v0.6_val_8.pkl.bbox.json'
ann_fn = '/home/admin/jupyter/wei/mmdetection-master/data/coco/annotations/instances_val2017_11cls.json'
root = 'jsons_from_v0.7/'
gt_path = root+'gt_cls11map6_val2017.json'
res_path = root+'res_cls11map6_v0.6.json'

gt_11cls = json.load(open(ann_fn,'r'))
res_11cls = json.load(open(res_fn,'r'))

for i in range(len(res_11cls)):
    cat_id = res_11cls[i]['category_id']
    res_11cls[i]['category_id'] = cat_id11map6[cat_id-1]

json.dump(cls11map6(gt_11cls), open(gt_path, 'w'), ensure_ascii=False, indent=2)
json.dump(res_11cls, open(res_path, 'w'), ensure_ascii=False, indent=2)

coco = COCO(gt_path)
coco_dets = coco.loadRes(res_path)

cocoEval = COCOeval(coco, coco_dets, 'bbox')
img_ids = coco.getImgIds()
cocoEval.params.imgIds = img_ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

table_classwise(cocoEval, coco)


