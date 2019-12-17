import kfbReader
import json
import numpy as np
import os
from tqdm import tqdm
import cv2
import multiprocessing as mp

def get_xylist(dataIds, W, H):
    if W < 1024:
        outlier = int(W/2)
    else:
        outlier = int(512)
    xys = {}
    roi_num = 0
    for dataId in dataIds:
        ress = json.load(open(test_path+dataId+'.json','r'))
        rois = []
        roiL = 0
        for res in ress:
            if res['class'] == 'roi':
                rois.append(res)
                roiL += 1
        roi_num += roiL
        if roiL == 0: continue
        xys[dataId] = []
        for roi in rois:
            width = int(roi['w'])
            height = int(roi['h'])
            roixmin = int(roi['x'])
            roiymin = int(roi['y'])
            roixmax = roixmin+width
            roiymax = roiymin+height
            ## 
            w = W
            h = H
            if w <= width:                                                         ## 正常的x方向
                xs = list(range(roixmin,roixmax,w-outlier))
            if h <= height:                                                         ## 正常的y方向
                ys = list(range(roiymin,roiymax,h-outlier))
            if w > width and h > height:                                    ## xy方向均不正常，取小的roi边长为步长
                if width >= height:
                    xs = list(range(roixmin,roixmax,height-outlier))
                    ys = [roiymin]
                    h = height
                    w = h
                else:
                    ys = list(range(roiymin,roiymax,width-outlier))
                    xs = [roixmin]
                    w = width
                    h = w
            elif w > width:                                                         ## x方向不正常 y正常
                xs = [roixmin]
                w = width
                h = w
            elif h > height:                                                         ## y方向不正常 x正常
                ys = [roiymin]
                h = height
                w = h
            ## 
            for ii in range(len(xs)):
                flag_x = 0
                xmin = xs[ii]
                xmax = xmin+w
                if xmax > roixmax:
                    xmax = roixmax
                    xmin = roixmax - h
                    flag_x = 1
                for jj in range(len(ys)):
                    flag_y = 0
                    ymin = ys[jj]
                    ymax = ymin+h
                    if ymax > roiymax:
                        ymax = roiymax
                        ymin = roiymax - h
                        flag_y = 1
                    xys[dataId].append([xmin,ymin,xmax,ymax])
                    if flag_y == 1:
                        break
                if flag_x == 1:
                    break
    return xys


def gen_topleft(dataIds, idi, topleft, imgid2name, w, h):
    xys = get_xylist(dataIds, w, h)
    for dataId in dataIds:
        slide = kfbReader.reader()
        kfbReader.reader.ReadInfo(slide, test_path + dataId + '.kfb', 20, True)
        xy = xys[dataId]
        for xmin,ymin,xmax,ymax in xy:
            imgid = idi
            idi += 1
            w = xmax-xmin
            h = ymax-ymin
            topleft[imgid] = [xmin, ymin, w, h]
            imgid2name[imgid] = dataId + '_' + str(xmin) + '_' + str(ymin) + '_' + str(w) + '_' + str(h) + '.jpg'

    return idi, topleft, imgid2name     


def gen_imgs(name, test_path, dir, imgids, topleft, imgid2name):
    
    slide = kfbReader.reader()
    pbar = tqdm(total = len(imgids))
    pbar.set_description(name+': Croping samples...')
    for imgid in imgids:
        pbar.update(1)
        tlx, tly, rw, rh = topleft[imgid]
        file_name = imgid2name[imgid]
        first_id = file_name.find('_')
        dataId = file_name[:first_id]
        kfbReader.reader.ReadInfo(slide, test_path + dataId + '.kfb', 20, False)
        if rw != 1024 or rh != 1024:
            simg = cv2.resize(slide.ReadRoi(tlx,tly,rw,rh,20),(1024,1024))
        else:
            simg = slide.ReadRoi(tlx,tly,rw,rh,20)
        if not cv2.imwrite(dir + file_name, simg): 
            print(dir + file_name + ' is existed ! Will be removed...')
            os.remove(dir + file_name)
            cv2.imwrite(dir + file_name, simg)
    
class myProcess(mp.Process):
    def __init__(self, threadID, name, test_path, dir, imgids, topleft, imgid2names):
        mp.Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.imgids = imgids
        self.topleft = topleft
        self.imgid2names = imgid2names
        self.test_path = test_path
        self.dir = dir
    def run(self):
        print ("Start: " + self.name)
        gen_imgs(self.name, self.test_path, self.dir, self.imgids, self.topleft, self.imgid2names)
        print ("End: " + self.name)


slide = kfbReader.reader()
tianchiCom = '/home/admin/jupyter/Data/'
test_path = tianchiCom + 'test/'
info = "spytensor created"
license = ["license"]
# categories = [{
#     'id': 1,
#     'name': 'ASC-H'
# },{
#     'id': 2,
#     'name': 'ASC-US'
# },{
#     'id': 3,
#     'name': 'HSIL'
# },{
#     'id': 4,
#     'name': 'LSIL'
# },{
#     'id': 5,
#     'name': 'Candida'
# },{
#     'id': 6,
#     'name': 'Trichomonas'
# }]
categories = [{
    'id': 1,
    'name': 'ASC-H1'
    },{
    'id': 2,
    'name': 'ASC-H2'
    },{
    'id': 3,
    'name': 'ASC-H3'
    },{
    'id': 4,
    'name': 'ASC-US1'
    },{
    'id': 5,
    'name': 'ASC-US2'
    },{
    'id': 6,
    'name': 'ASC-US3'
    },{
    'id': 7,
    'name': 'HSIL1'
    },{
    'id': 8,
    'name': 'HSIL2'
    },{
    'id': 9,
    'name': 'HSIL3'
    },{
    'id': 10,
    'name': 'LSIL1'
    },{
    'id': 11,
    'name': 'LSIL2'
    },{
    'id': 12,
    'name': 'LSIL3'
    },{
    'id': 13,
    'name': 'Candida1'
    },{
    'id': 14,
    'name': 'Candida2'
    },{
    'id': 15,
    'name': 'Candida3'
    },{
    'id': 16,
    'name': 'Candida4'
    },{
    'id': 17,
    'name': 'Trichomonas1'
    },{
    'id': 18,
    'name': 'Trichomonas2'
    }]
scale_r = [4, 1, 0.4, 0.1]
W=[512, 1024, 2560, 10240]  # 1024 * scale_r
H=W

root_dir = '/home/admin/jupyter/wei/mmdetection-master/data/coco/'
test_dir = root_dir + 'test2017_with512/'
ann_dir = root_dir + 'annotations/'
if not(os.path.exists(root_dir)):
    os.mkdir(root_dir)
if not(os.path.exists(test_dir)):
    os.mkdir(test_dir)
if not(os.path.exists(ann_dir)):
    os.mkdir(ann_dir)

for r,d,fs in os.walk(test_path):
    break
dataIds = []
for f in fs:
    if f[-1] == 'b':
        dataIds.append(f[:-4])

idi = 0
topleft = {}
imgid2name = {}
################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[0], H[0])
################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[1], H[1])
################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[2], H[2])
################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[3], H[3])
################################################################################

test_topleft, test_imgid2name = topleft, imgid2name 
print()
print('Num of test Images: '+str(len(test_topleft)))
print()

###############
## test2017 ##
###############

data_test = {}
data_test['info'] = info
data_test['license'] = license
data_test['annotations'] = []
images = []
pbar = tqdm(total = len(test_topleft))
pbar.set_description('Generating json...')
i = 0
for imgid in test_topleft:
    pbar.update(1)
    tlx, tly, rw, rh = test_topleft[imgid]
    file_name = test_imgid2name[imgid]
    images.append({
            "height": rh,
            "width": rw,
            "id": imgid,
            "file_name": file_name
        })
data_test['images'] = images
data_test['categories'] = categories
with open(ann_dir + 'instances_test2017.json','w') as f:
    json.dump(data_test, f)

cp = [i for i in range(0,len(test_topleft),int(len(test_topleft)/8))]
cp[-1] = len(test_topleft)
threads = []

for i in range(8):
    th = myProcess(i+1, "Process-"+str(i+1), test_path, test_dir, list(test_topleft.keys())[cp[i]:cp[i+1]], test_topleft, test_imgid2name)
    threads.append(th)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()