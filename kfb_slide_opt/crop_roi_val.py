import kfbReader
import json
import numpy as np
import os
from tqdm import tqdm
import cv2
import multiprocessing as mp
import pandas as pd


def get_scale_part_11cls(side_len):

    if side_len <= 600:
        return '1'
    else:
        return '3'
    return ''


def get_scale_part_16cls(side_len):

    if side_len <= 100:
        return '1'
    elif 100 <= side_len <= 600:
        return '2'
    else:
        return '3'
    return ''


def gen_csv(dataIds, pos_path, save_path, idi, topleft, imgid2name):
    labels_num = 0
    slide = kfbReader.reader()
    paths=[]
    x_min=[]
    y_min=[]
    x_max=[]
    y_max=[]
    cls=[]
    image_id=[]
    pbar = tqdm(total = idi)
    pbar.set_description('Checking labels')

    for img_id in range(idi):
        pbar.update(1)
        img_fn = imgid2name[img_id]
        tlx, tly, tlw, tlh = topleft[img_id]
        dataId = img_fn[:img_fn.find('_')]
        scale_x = tlw/1024
        scale_y = tlh/1024
        kfbReader.reader.ReadInfo(slide, pos_path + dataId + '.kfb', 20, True)
        labels = json.load(open(pos_path + dataId + '.json','r'))
        poss=[]

        for label in labels:
            if label['class'] == 'roi': continue
            poss.append(label)

        n_labels = 0

        for pos_ii in poss:
            pxi=pos_ii['x'];pyi=pos_ii['y'];pwi=pos_ii['w'];phi=pos_ii['h'] 
            if pxi>=tlx and pyi>=tly and pxi+pwi<=tlx+tlw and pyi+phi<=tly+tlh: #inner label
                paths.append(save_path+img_fn)
                xxmin=int((pxi-tlx)/scale_x)
                yymin=int((pyi-tly)/scale_y)
                xxmax=int((pxi-tlx+pwi)/scale_x)
                yymax=int((pyi-tly+phi)/scale_y)
                x_min.append(xxmin)
                y_min.append(yymin)
                x_max.append(xxmax)
                y_max.append(yymax)
                image_id.append(img_id)
                cls.append(pos_ii['class']) 
                n_labels += 1

        labels_num += n_labels

    data = [[paths[i], x_min[i], y_min[i], x_max[i],y_max[i],cls[i],image_id[i]] for i in range(len(paths))]
    print(len(data))
    csv_fn = "/home/admin/jupyter/zxy/label_roival.csv"
    print('Output CSV with ' + str(labels_num) + ' labels in ' + csv_fn)
    col = ['path', 'xmin', 'ymin','xmax','ymax','cls','img_id']
    df = pd.DataFrame(data,columns=col)
    df.to_csv(csv_fn,index=False)        

    return csv_fn

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

def csv2coco(csv_fn, classname_to_id):

    total_csv_annotations = {}
    annotations = pd.read_csv(csv_fn, header=None).values
    anns = []

    for annotation in annotations[1:]:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value

    ann_id = 0

    for key in total_csv_annotations.keys():
        line = total_csv_annotations[key][0]
        min_x, min_y, max_x, max_y = [int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        w = max_x - min_x
        h = max_y - min_y
        if len(classname_to_id) == 6:
            scale_part = ''
        elif len(classname_to_id) == 11:
            scale_part = get_scale_part_11cls(max([w,h]))
        elif len(classname_to_id) == 16:
            scale_part = get_scale_part_16cls(max([w,h]))
        
        annotation = {}
        annotation['id'] = ann_id
        annotation['image_id'] = int(line[5])
        annotation['category_id'] = int(classname_to_id[line[4]+scale_part])
        annotation['segmentation'] = [min_x, min_y, min_x, min_y + 0.5 * h, min_x, max_y, min_x + 0.5 * w, max_y, max_x, max_y, max_x, max_y - 0.5 * h, max_x, min_y, max_x - 0.5 * w, min_y]
        annotation['bbox'] = [min_x, min_y, max_x, max_y]
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        anns.append(annotation)
        ann_id += 1

    return anns


def gen_json(dir, annotations, imgid2name, categories, json_fn):

    info = "spytensor created"
    license = ["license"]
    data_test = {}
    data_test['info'] = info
    data_test['license'] = license
    data_test['annotations'] = annotations
    images = []
    pbar = tqdm(total = len(imgid2name))
    pbar.set_description('Generating json to '+json_fn)
    
    for imgid in imgid2name:
        pbar.update(1)
        file_name = imgid2name[imgid]
        images.append({
                "height": 1024,
                "width": 1024,
                "id": imgid,
                "file_name": file_name
            })
    data_test['images'] = images
    data_test['categories'] = categories
    with open(dir + json_fn,'w') as f:
        json.dump(data_test, f)

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

classname_to_id_6cls = {'ASC-H':1,'ASC-US':2,'HSIL':3,'LSIL':4,'Candida':5,'Trichomonas':6}
classname_to_id_11cls = {'ASC-H1':1,'ASC-H3':2,'ASC-US1':3,'ASC-US3':4,'HSIL1':5,'HSIL3':6,'LSIL1':7,'LSIL3':8,'Candida1':9,'Candida3':10,'Trichomonas1':11}
classname_to_id_16cls = {'ASC-H1':1,'ASC-H2':2,'ASC-H3':3,'ASC-US1':4,'ASC-US2':5,'ASC-US3':6,'HSIL1':7,'HSIL2':8,'HSIL3':9,'LSIL1':10,'LSIL2':11,'LSIL3':12,'Candida1':13,'Candida2':14,'Candida3':15,'Trichomonas1':16}
categories_6cls = [{'id': 1,'name': 'ASC-H'},{'id': 2,'name': 'ASC-US'},{'id': 3,'name': 'HSIL'},{'id': 4,'name': 'LSIL'},{'id': 5,'name': 'Candida'},{'id': 6,'name': 'Trichomonas'}]
categories_11cls = [{'id': 1,'name': 'ASC-H1'},{'id': 2,'name': 'ASC-H3'},{'id': 3,'name': 'ASC-US1'},{'id': 4,'name': 'ASC-US3'},{'id': 5,'name': 'HSIL1'},{'id': 6,'name': 'HSIL3'},{'id': 7,'name': 'LSIL1'},{'id': 8,'name': 'LSIL3'},{'id': 9,'name': 'Candida1'},{'id': 10,'name': 'Candida3'},{'id': 11,'name': 'Trichomonas1'}]
categories_16cls = [{'id': 1,'name': 'ASC-H1'},{'id': 2,'name': 'ASC-H2'},{'id': 3,'name': 'ASC-H3'},{'id': 4,'name': 'ASC-US1'},{'id': 5,'name': 'ASC-US2'},{'id': 6,'name': 'ASC-US3'},{'id': 7,'name': 'HSIL1'},{'id': 8,'name': 'HSIL2'},{'id': 9,'name': 'HSIL3'},{'id': 10,'name': 'LSIL1'},{'id': 11,'name': 'LSIL2'},{'id': 12,'name': 'LSIL3'},{'id': 13,'name': 'Candida1'},{'id': 14,'name': 'Candida2'},{'id': 15,'name': 'Candida3'},{'id': 16,'name': 'Trichomonas1'}]
scale_r = [2, 1, 0.4, 0.1]
W=[512, 1024, 2560, 10240]  # 1024 * scale_r
H=W

tianchiCom = '/home/admin/jupyter/Data/'
test_path = tianchiCom + 'train/'
root_dir = '/home/admin/jupyter/wei/mmdetection-master/data/coco/'
test_dir = root_dir + 'roival2017/'
ann_dir = root_dir + 'annotations/'
if not(os.path.exists(root_dir)):
    os.mkdir(root_dir)
if not(os.path.exists(test_dir)):
    os.mkdir(test_dir)
if not(os.path.exists(ann_dir)):
    os.mkdir(ann_dir)

for r,d,fs in os.walk(test_path):
    break
fns = []
for f in fs:
    if f[-1] == 'n':
        fns.append(f[:-5])
step_length = len(fns)/72
dataIds = []
for i in range(0,len(fns),int(step_length)):
    dataIds.append(fns[i])

idi = 0
topleft = {}
imgid2name = {}
###########################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[0], H[0])
###########################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[1], H[1])
###########################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[2], H[2])
###########################################################################################
idi, topleft, imgid2name = gen_topleft(dataIds, idi, topleft, imgid2name, W[3], H[3])
###########################################################################################

test_topleft, test_imgid2name = topleft, imgid2name 
print()
print('Num of test Images: '+str(len(test_topleft)))
print()
csv_fn = gen_csv(dataIds, test_path, test_dir, idi, test_topleft, test_imgid2name)
#csv_fn = '/home/admin/jupyter/zxy/label_roival.csv'
json_fn_6cls = 'instances_roival2017_6cls.json'
json_fn_11cls = 'instances_roival2017_11cls.json'
json_fn_16cls = 'instances_roival2017_16cls.json'
###########################################################################################
annotations_6cls = csv2coco(csv_fn, classname_to_id_6cls)
print('6cls: '+str(len(annotations_6cls)))
gen_json(ann_dir, annotations_6cls, test_imgid2name, categories_6cls, json_fn_6cls)
###########################################################################################
annotations_11cls = csv2coco(csv_fn, classname_to_id_11cls)
print('11cls: '+str(len(annotations_11cls)))
gen_json(ann_dir, annotations_11cls, test_imgid2name, categories_11cls, json_fn_11cls)
###########################################################################################
annotations_16cls = csv2coco(csv_fn, classname_to_id_16cls)
print('16cls: '+str(len(annotations_16cls)))
gen_json(ann_dir, annotations_16cls, test_imgid2name, categories_16cls, json_fn_16cls)
###########################################################################################

#################
## roival2017 ##
#################
exit()
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