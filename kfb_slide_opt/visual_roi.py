import json
import numpy as np
import kfbReader
import sys
from tqdm import tqdm
import cv2
import os

fs = []
demo_dir = 'roi_illustrations/'
pos_path = '/home/admin/jupyter/Data/train/'
for root, dirs, files in os.walk(pos_path):
    for file in files:
        if file[-1] == 'n':
            fs.append(file[:-5])

for dataId in fs:
    rois = []
    poss = []
    labels = json.load(open(pos_path + dataId + '.json','r'))
    slide = kfbReader.reader()
    kfbReader.reader.ReadInfo(slide, pos_path + dataId + '.kfb', 20, True)
    for label in labels:
        if label['class'] != 'roi': continue
        rois.append(label)
    for label in labels:
        if label['class'] == 'roi': continue
        poss.append(label)
    for roi in rois:
        xx = np.random.uniform(0,1)
        if xx < 0.9: continue
        sx = int(roi['x'])
        sy = int(roi['y'])
        sw = int(roi['w'])
        sh = int(roi['h'])

        img = slide.ReadRoi(sx,sy,sw,sh,20)
        inners = []
        for pos in poss:
            
            px = int(pos['x'])
            py = int(pos['y'])
            pw = int(pos['w'])
            ph = int(pos['h'])
            cls = pos['class']
            if px>=sx and py>=sy and px+pw<=sx+sw and py+ph<=sy+sh:
                rx = int(px-sx)
                ry = int(py-sy)
                cv2.rectangle(img,(rx,ry),(rx+pw,ry+ph),(0,0,255),2)
                cv2.putText(img, cls, (int(rx),int(ry)), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)    
                inners.append(pos)
                print(pos)
        cv2.imwrite(demo_dir+dataId+'_zxy'+str(sx)+'_'+str(sy)+'_roi.jpg',img)
        print(len(inners))
        input()
