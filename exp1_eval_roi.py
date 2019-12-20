import json
import os
import numpy as np
import itertools
from terminaltables import AsciiTable

def dict2ndarray(dict):
    return np.array([float(dict['x']),float(dict['y']),float(dict['x'])+float(dict['w']),float(dict['y'])+float(dict['h'])])

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

root_path = '/home/admin/jupyter/wei/mmdetection-master/res_for_submit/'
tianchiCom = '/home/admin/jupyter/Data/'
test_path = tianchiCom + 'train/'
iou_threshold = 0.5

load_path = 'f_r50_v0.7_roival_7_nmsSeparative0.5/'

for r,d,fs in os.walk(test_path):
    break
fns = []
for f in fs:
    if f[-1] == 'n':
        fns.append(f[:-5])
step_length = len(fns)/20
dataIds = []
for i in range(0,len(fns),int(step_length)):
    dataIds.append(fns[i])


for root, dirs, files in os.walk(root_path+load_path):
    break


total_ap = {}

for file in files:
    res = {}
    gts = {}

    for gt in json.load(open(test_path+file,'r')):
        if gt['class'] != 'roi':
            if gt['class'] not in gts: gts[gt['class']] = []
            gts[gt['class']].append(dict2ndarray(gt))

    for rec in json.load(open(root_path+load_path+file,'r')):
        if rec['class'] not in res: res[rec['class']] = []
        res[rec['class']].append(np.append(dict2ndarray(rec), float(rec['p'])))

    average_precisions = {}
    for cls in res:
        if cls not in gts: gts[cls] = []
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        detections = np.array(res[cls])
        annotations = np.array(gts[cls])
        detected_annotations = []
        num_annotations = len(annotations)

        for d in detections:
            scores = np.append(scores, d[4])
            if len(annotations) == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue
            overlaps = compute_overlap(np.expand_dims(d[:4], axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
            # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[cls] = 0, 0
            continue
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[cls] = average_precision, num_annotations
        if cls not in total_ap: total_ap[cls] = []
        total_ap[cls].append(average_precisions[cls])

    results_per_category = []
    mAPS = []
    for cls in average_precisions:
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        ap = average_precisions[cls][0]
        results_per_category.append(('{}'.format(cls+str(average_precisions[cls][1])),'{:0.3f}'.format(float(ap * 100))))
        mAPS.append(ap)

    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    headers = ['category', 'AP'] * (N_COLS // 2)
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print(table.table)
    print('{}'.format(file)+' mAP: {}'.format(np.mean(mAPS)))

results_per_category = []
mAP = []
ann_num = 0
for cls in total_ap:
    # area range index 0: all area ranges
    # max dets index -1: typically 100 per image
    total_AP = np.array(total_ap[cls])
    ap = np.mean(total_AP[:,0])
    results_per_category.append(('{}'.format(cls+str(np.sum(total_AP[:,1]))),'{:0.3f}'.format(float(ap * 100))))
    mAP.append(ap)
    ann_num += np.sum(total_AP[:,1])
print('Contains '+str(int(ann_num))+' ground truth')
N_COLS = min(6, len(results_per_category) * 2)
results_flatten = list(itertools.chain(*results_per_category))
headers = ['category', 'AP'] * (N_COLS // 2)
results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
table_data = [headers]
table_data += [result for result in results_2d]
table = AsciiTable(table_data)
print(table.table)
print('Total mAP: {}'.format(np.mean(mAP)))
