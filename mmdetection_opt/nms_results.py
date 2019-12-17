import json
import numpy as np
import sys
from tqdm import tqdm
import os

def nms(x1, y1, x2, y2, scores, thresh):  
    """Pure Python NMS baseline."""  

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]    
    keep = []  
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
        # indd = np.where(ovr > thresh)[0]  
        # print(str(i)+' <- '+str(indd))
        order = order[inds + 1]  
  
    return keep

def soft_nms(x1, y1, x2, y2, scores, thresh, sigma2=0.5):  
    """Pure Python NMS baseline."""  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
    nmsed_scrs = scores
    keep = []  
    threshed = []
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
        greater = np.where(ovr > thresh)[0]  
        nmsed_scrs[order[inds + 1]] *= 1
        nmsed_scrs[order[greater + 1]] *= np.exp(-(ovr[greater]**2)/sigma2)
        threshed.extend(order[greater + 1])
        order = order[inds + 1] 

    k = np.where(nmsed_scrs[threshed] > 0.05)[0]
    keep.extend(np.array(threshed)[k])

    return keep

######################################################################################
root_path = '/home/admin/jupyter/wei/mmdetection-master/res_for_submit/'
tianchiCom = '/home/admin/jupyter/Data/'
test_path = tianchiCom + 'test/'
thres = 0.5
load_path = 'f_fpn_ep7_v0.6/'
save_path = 'f_fpn_ep7_v0.6_nms'+str(thres)+'/'

if save_path[-1] != '/': save_path += '/'
for r,d,fs in os.walk(test_path):
    break
dataIds = []
for f in fs:
    if f[-1] == 'b':
        dataIds.append(f[:-4])
for root, dirs, files in os.walk(root_path+load_path):
    break
mw = 0
if not(os.path.exists(root_path+save_path)):
    os.mkdir(root_path+save_path)
for file in files:
    dataIds.remove(file[:-5])
    with open(root+'/'+file,'rb+') as load_f:
        npyf = json.load(load_f)
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        scores = []
        for rec in npyf:
            x=rec['x']
            y=rec['y']
            w=rec['w']
            h=rec['h']
            p=rec['p']
            x1.append(x)
            y1.append(y)
            x2.append(x+w)
            y2.append(y+h)
            scores.append(p)
            if w > mw: mw = w
        keep_id = soft_nms(np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores), thresh=thres)
        save_res = []
        for i in keep_id:
            save_res.append(npyf[i])
        json_name = root_path+save_path+file
        with open(json_name,'w',encoding='utf-8') as jsonF:
            json.dump(save_res, jsonF, ensure_ascii=False)
            print()
            print(file+': '+str(len(npyf))+' -> '+str(len(keep_id)))
print('max w:'+str(mw))
print(str(len(dataIds))+' slide are empty')

for empJson in dataIds:
    json_name = root_path+save_path+empJson+'.json'
    with open(json_name,'w',encoding='utf-8') as jsonF:
        json.dump([], jsonF, ensure_ascii=False)
        print(json_name)