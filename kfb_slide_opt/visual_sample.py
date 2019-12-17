import os
import json
import numpy as np
import pandas as pd
import cv2
import os

csv_file = "/home/admin/jupyter/zxy/label.csv"
csv_file_1213 = "/home/admin/jupyter/zxy/label_1213.csv"
image_dir = "/home/admin/jupyter/zxy/sample_pos_data_1024/"
demo_dir = 'demo_puts_all_label_1213/'
# 整合csv格式标注文件
total_csv_annotations = {}
annotations = pd.read_csv(csv_file, header=None).values
annotations_1213 = pd.read_csv(csv_file_1213, header=None).values
key_0 = []
key_01 = []
key_1 = []
key_2 = []
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key != 'path':
        if float(key[key.rfind('_')+1:-4]) == 1 or float(key[key.rfind('_')+1:-4]) == 0.1: continue
        if float(key[key.rfind('_')+1:-4]) < 1: 
            key_0.append(key)
        if float(key[key.rfind('_')+1:-4]) == 0.1: key_01.append(key)
        if float(key[key.rfind('_')+1:-4]) == 1: key_1.append(key)
        if float(key[key.rfind('_')+1:-4]) > 1: key_2.append(key)
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
    else:
        total_csv_annotations[key] = value
for annotation in annotations_1213:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key != 'path':
        if float(key[key.rfind('_')+1:-4]) < 1: 
            key_0.append(key)
        if float(key[key.rfind('_')+1:-4]) == 0.1: key_01.append(key)
        if float(key[key.rfind('_')+1:-4]) == 1: key_1.append(key)
        if float(key[key.rfind('_')+1:-4]) > 1: key_2.append(key)
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
    else:
        total_csv_annotations[key] = value
# 按照键值划分数
total_keys = list(total_csv_annotations.keys())[1:]
################################################################################
for _ in range(10):
    demo = total_keys[np.random.randint(0,len(total_keys)-1)]
    demo = key_0[np.random.randint(0,len(key_0)-1)]
    img = cv2.imread(image_dir+demo)
    for ann in total_csv_annotations[demo]:
    #ann = [0]
        print(ann)
        cv2.rectangle(img,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),(255,0,0))
        cv2.putText(img, ann[4], (int(ann[0]),int(ann[1])), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)        
    cv2.imwrite(demo_dir+ann[4]+demo,img)
    input()
for _ in range(10):
    demo = total_keys[np.random.randint(0,len(total_keys)-1)]
    demo = key_01[np.random.randint(0,len(key_01)-1)]
    img = cv2.imread(image_dir+demo)
    for ann in total_csv_annotations[demo]:
    #ann = [0]
        print(ann)
        cv2.rectangle(img,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),(255,0,0))
        cv2.putText(img, ann[4], (int(ann[0]),int(ann[1])), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)        
    cv2.imwrite(demo_dir+ann[4]+demo,img)
    input()
for _ in range(10):
    demo = total_keys[np.random.randint(0,len(total_keys)-1)]
    demo = key_1[np.random.randint(0,len(key_1)-1)]
    img = cv2.imread(image_dir+demo)
    for ann in total_csv_annotations[demo]:
    #ann = [0]
        print(ann)
        cv2.rectangle(img,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),(255,0,0))
        cv2.putText(img, ann[4], (int(ann[0]),int(ann[1])), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)        
    cv2.imwrite(demo_dir+ann[4]+demo,img)
    input()
for _ in range(10):
    demo = total_keys[np.random.randint(0,len(total_keys)-1)]
    demo = key_2[np.random.randint(0,len(key_2)-1)]
    img = cv2.imread(image_dir+demo)
    for ann in total_csv_annotations[demo]:
    #ann = [0]
        print(ann)
        cv2.rectangle(img,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),(255,0,0))
        cv2.putText(img, ann[4], (int(ann[0]),int(ann[1])), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)        
    cv2.imwrite(demo_dir+ann[4]+demo,img)
    input()