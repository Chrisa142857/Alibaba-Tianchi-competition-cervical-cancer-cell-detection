import json
import numpy as np
import kfbReader
import sys
from tqdm import tqdm
import cv2
import os

def nms(x1, y1, x2, y2, scores, thresh):
    """Pure Python NMS baseline."""
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    ii = 0
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        indd = np.where(ovr > thresh)[0]
        # print('No. '+str(ii+1)+': '+str(i)+' <- '+str(order[indd + 1]))
        # ii += 1
        order = order[inds + 1]

    return keep

demo_dir = 'demo_f_fpn_v0.6/'
names = []
slide_path = []
test_dir = '/home/admin/jupyter/Data/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        slide_path.append(root+'/'+file)
        names.append(file)

rr = {}
i = 0
for one_path in slide_path:
    name = names[i]
    rr[name] = one_path
    i += 1
slideID = '7378.json'
slide861 = '/home/admin/jupyter/wei/mmdetection-master/res_for_submit/f_fpn_ep7_v0.6/'+slideID
with open(slide861,'r') as load_f:
    npyf = json.load(load_f)
rois = json.load(open(test_dir+'/'+slideID,'r'))
slide = kfbReader.reader()
kfbReader.reader.ReadInfo(slide, rr[slideID[:-5]+'.kfb'], 20, True)

for roi in rois:
    sx = roi['x']
    sy = roi['y']
    sw =  roi['w']
    sh =  roi['h']

    img = slide.ReadRoi(sx,sy,sw,sh,20)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scores = []
    img0 = img.copy()
    for rec in npyf:
        x=rec['x']
        y=rec['y']
        w=int(rec['w'])
        h=int(rec['h'])
        p=rec['p']
        x1.append(x)
        y1.append(y)
        x2.append(x+w)
        y2.append(y+h)
        scores.append(p)
        if x>=sx and y>=sy and x+w<=sx+sw and y+h<=sy+sh:
            rx = int(x-sx)
            ry = int(y-sy)
            cv2.rectangle(img,(rx,ry),(rx+w,ry+h),(0,0,255),2)
            cv2.putText(img, rec['class']+':'+str(p), (int(rx),int(ry)), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)        
    cv2.imwrite(demo_dir+slideID[:-5]+'_'+str(sx)+'_'+str(sy)+'.jpg',img)


    keep1 = nms(np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores), thresh=0.3)      
    print(len(npyf),len(keep1))
    for i in keep1:
        rec = npyf[i]
        x=rec['x']
        y=rec['y']
        w=int(rec['w'])
        h=int(rec['h'])
        p=rec['p']
        if x>=sx and y>=sy and x+w<=sx+sw and y+h<=sy+sh:
            rx = int(x-sx)
            ry = int(y-sy)
            cv2.rectangle(img0,(rx,ry),(rx+w,ry+h),(0,0,255),2)
            cv2.putText(img0, rec['class']+':'+str(p), (int(rx),int(ry)), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)       
    cv2.imwrite(demo_dir+slideID[:-5]+'_'+str(sx)+'_'+str(sy)+'_nmsed.jpg',img0)

    input()
exit(0)