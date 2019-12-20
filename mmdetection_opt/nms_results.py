import json
import numpy as np
import sys
from tqdm import tqdm
import os

def nms_allcls(x1, y1, x2, y2, scores, thresh): 
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

def nms(x1, y1, x2, y2, scores, cls, thresh):  
    """Pure Python NMS baseline."""  

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]    
    keep = []
    while order.size > 0:  
        i = order[0]
        clas = cls[i]
        keep.append(i)
        in_cls_order = np.array([ii for ii in order[1:] if cls[ii] == clas])
        if in_cls_order.size == 0: 
            order = order[1:]
            continue
        xx1 = np.maximum(x1[i], x1[in_cls_order])  
        yy1 = np.maximum(y1[i], y1[in_cls_order])  
        xx2 = np.minimum(x2[i], x2[in_cls_order])  
        yy2 = np.minimum(y2[i], y2[in_cls_order])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[in_cls_order] - inter)  
        
        # inds = np.where(ovr <= thresh)[0]  
        indd = np.where(ovr > thresh)[0]
        order = np.setdiff1d(order,in_cls_order[indd])[1:]
        # print(str(i)+' <- '+str(indd))
        # order = order[inds + 1]  
  
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
test_path = tianchiCom + 'train/'
thres = 0.5
load_path = 'f_r50_v0.6_roival_7/'
save_path = 'f_r50_v0.6_roival_7_nmsSeparative'+str(thres)+'/'

if save_path[-1] != '/': save_path += '/'
#for r,d,fs in os.walk(test_path):
#    break
#dataIds = []
#for f in fs:
#    if f[-1] == 'b':
#        dataIds.append(f[:-4])

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
mw = 0
if not(os.path.exists(root_path+save_path)):
    os.mkdir(root_path+save_path)
for file in files:
    dataIds.remove(file[:-5])
    load_f = open(root+'/'+file,'rb+')
    npyf = json.load(load_f)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scores = []
    cls = []
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
        cls.append(rec['class'])
        if w > mw: mw = w
    keep_id = nms_allcls(np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores), thresh=thres)
    save_res = []
    for i in keep_id:
        if npyf[i]['p'] < 0.1: continue
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