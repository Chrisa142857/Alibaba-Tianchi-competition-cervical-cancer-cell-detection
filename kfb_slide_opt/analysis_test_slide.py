import kfbReader
import json
import numpy as np
import os
from tqdm import tqdm
import random
import cv2


slide = kfbReader.reader()
tianchiCom = '/home/admin/jupyter/Data/'
test_path = tianchiCom + 'test/'
info = "spytensor created"
license = ["license"]
categories = [{
    'id': 1,
    'name': 'ASC-H'
},{
    'id': 2,
    'name': 'ASC-US'
},{
    'id': 3,
    'name': 'HSIL'
},{
    'id': 4,
    'name': 'LSIL'
},{
    'id': 5,
    'name': 'Candida'
},{
    'id': 6,
    'name': 'Trichomonas'
}]
W = 800
H = 800
tpt = 0.2   # val占比
root_dir = '/home/admin/jupyter/wei/mmdetection-master/data/coco/'
train_dir = root_dir + 'train2017/'
val_dir = root_dir + 'val2017/'
ann_dir = root_dir + 'annotations/'

for r,d,fs in os.walk(test_path):
    break

dataIds = []
for f in fs:
    if f[-1] == 'b':
        dataIds.append(f[:-4])

w_sizes = {}
h_sizes = {}

for dataId in dataIds:
    kfbReader.reader.ReadInfo(slide, test_path + dataId + '.kfb', 20, True)
    width = slide.getWidth()
    height = slide.getHeight()
    w_size_key = int(width / 10000)
    h_size_key = int(height / 10000)
    if w_size_key not in w_sizes: w_sizes[w_size_key] = 0
    if h_size_key not in h_sizes: h_sizes[h_size_key] = 0
    w_sizes[w_size_key] += 1
    h_sizes[h_size_key] += 1


for w_size_key in w_sizes: 
    strr = 'Num in w_size '+str(w_size_key*10000)+' ~ '+str((w_size_key+1)*10000)+' : '
    print(strr.ljust(50,' ')+str(w_sizes[w_size_key]))
    
for h_size_key in h_sizes: 
    strr = 'Num in h_size '+str(h_size_key*10000)+' ~ '+str((h_size_key+1)*10000)+' : '
    print(strr.ljust(50,' ')+str(h_sizes[h_size_key]))
