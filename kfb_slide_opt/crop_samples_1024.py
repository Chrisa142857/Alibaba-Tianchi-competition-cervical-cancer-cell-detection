import kfbReader
import json
import numpy as np
import os
from tqdm import tqdm
import random
import cv2
import multiprocessing as mp
import threading
import time
import math
import pandas as pd

demo_dir = 'sample_illustrations/'

def gen_csv():
    img_fns = {}
    sample_xywh = {}
    imgs_num = 0
    labels_num = 0
    slide = kfbReader.reader()
    paths=[]
    x_min=[]
    y_min=[]
    x_max=[]
    y_max=[]
    cls=[]
    scale_nums = [0,0,0,0]
    for dataId in dataIds:   
        sample_xywh[dataId] = []
        img_fns[dataId] = []
        poss=[]

        pbar.update(1)
        kfbReader.reader.ReadInfo(slide, pos_path + dataId + '.kfb', 20, True)
        width = slide.getWidth()
        height = slide.getHeight()
        labels = json.load(open(pos_path + dataId + '.json','r'))

        for label in labels:
            if label['class'] == 'roi': continue
            poss.append(label)
        
        for k,pos_i in enumerate(poss):
            gx = int(pos_i['x'])
            gy = int(pos_i['y'])
            gw = int(pos_i['w'])
            gh = int(pos_i['h'])
            # garea=int(math.sqrt(gw*gh))
            garea = max([gw,gh])
            #if garea<100 or garea>=600 and garea<1000:
            #    continue
            #if not garea>1000:
            #    continue

            for i in range(weight[categories[pos_i['class']]]):

                if garea<100:
                    scale_ratio=np.random.uniform(2, 6)
                    scale_nums[0] += 1
                elif garea>=600 and garea<1000:
                    scale_ratio=np.random.uniform(0.2, 0.6)
                    scale_nums[1] += 1
                elif garea>1000:
                    scale_ratio=0.1
                    scale_nums[2] += 1
                else:
                    scale_ratio=1.0
                    scale_nums[3] += 1

                sgx=int(gx); sgy=int(gy); sgw=int(gw); sgh=int(gh)
                sample_w=int(W/scale_ratio); sample_h=int(H/scale_ratio)
                rangx_max=sgx; rangx_min=sgx+sgw-sample_w
                rangy_max=sgy; rangy_min=sgy+sgh-sample_h
                randomx=random.randint(rangx_min,rangx_max)
                randomy=random.randint(rangy_min,rangy_max) 
                sample_x=int(randomx); sample_y=int(randomy)
                img_fn = str(dataId)+'_'+str(sample_x)+'_'+str(sample_y)+'_'+str(scale_ratio)+'.jpg'
                marksave_path = os.path.join(marksave_dir+img_path, img_fn)

                #xx = np.random.uniform(0,1)
                #if scale_ratio == 1 and pos_i['class'] != 'Candida' and xx >= 0.9:
                #    img = cv2.resize(slide.ReadRoi(sample_x,sample_y,sample_w,sample_h,20),(1024,1024))

                n_labels = 0

                for pos_ii in poss:
                    pxi=pos_ii['x'];pyi=pos_ii['y'];pwi=pos_ii['w'];phi=pos_ii['h'] 
                    if pxi>=sample_x and pyi>=sample_y and pxi+pwi<=sample_x+sample_w and pyi+phi<=sample_y+sample_h: #inner label
                        paths.append(marksave_path)
                        xxmin=int((pxi-sample_x)*scale_ratio)
                        yymin=int((pyi-sample_y)*scale_ratio)
                        xxmax=int((pxi-sample_x+pwi)*scale_ratio)
                        yymax=int((pyi-sample_y+phi)*scale_ratio)
                        x_min.append(xxmin)
                        y_min.append(yymin)
                        x_max.append(xxmax)
                        y_max.append(yymax)
                        cls.append(pos_ii['class']) 
                        n_labels += 1
 
                #if scale_ratio == 1 and pos_i['class'] != 'Candida' and xx >= 0.9:
                #    for i in range(len(paths)):
                #        if paths[i] == marksave_path:
                #            cv2.rectangle(img,(x_min[i],y_min[i]),(x_max[i],y_max[i]),(0,0,255),2)
                #            cv2.putText(img, cls[i], (x_min[i],y_min[i]), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1) 
                #            print([paths[i], x_min[i], y_min[i], x_max[i],y_max[i],cls[i]])
                #    cv2.imwrite(demo_dir+dataId+'_'+str(sample_x)+'_'+str(sample_y)+'_'+ str(scale_ratio) +'.jpg',img)
                #    print(n_labels)
                #    input() 

                labels_num += n_labels

                if n_labels != 0:
                    sample_xywh[dataId].append([sample_x,sample_y,sample_w,sample_h])
                    img_fns[dataId].append(img_fn)
                    imgs_num += 1
                    
    data = [[paths[i], x_min[i], y_min[i], x_max[i],y_max[i],cls[i]] for i in range(len(paths))]
    print(len(data),imgs_num)
    csv_fn = "/home/admin/jupyter/zxy/label_2393&8484.csv"
    print('Output CSV with ' + str(imgs_num) + ' imgs and ' + str(labels_num) + ' labels in ' + csv_fn)
    print('2~6 0.2~0.6 0.1 1')
    print(scale_nums)
    col = ['path', 'xmin', 'ymin','xmax','ymax','cls']
    df = pd.DataFrame(data,columns=col)
    df.to_csv(csv_fn,index=False)        

    return sample_xywh, img_fns
                
def gen_imgs(name, dataIds, sample_xywh, img_fns):

    bar = tqdm(total = len(dataIds))
    bar.set_description(name + ' Croping...')
    
    slide = kfbReader.reader()
    for dataId in dataIds:
        bar.update(1)
        kfbReader.reader.ReadInfo(slide, pos_path + dataId + '.kfb', 20, True)
        for i in range(len(sample_xywh[dataId])):
            sample_x,sample_y,sample_w,sample_h = sample_xywh[dataId][i]
            img_fn = img_fns[dataId][i]
            scale_sample_img=cv2.resize(slide.ReadRoi(sample_x,sample_y,sample_w,sample_h,20), (W, H))
            marksave_path = os.path.join(img_dir, img_fn)
            if not(cv2.imwrite(marksave_path, scale_sample_img)):
                print(marksave_path + ' is existed ! Will be removed...')
                os.remove(marksave_path)
                cv2.imwrite(marksave_path, scale_sample_img)


class myProcess(mp.Process):
    def __init__(self, threadID, name, dataIds, sample_xywh, img_fns):
        mp.Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataIds = dataIds
        self.sample_xywh = sample_xywh
        self.img_fns = img_fns
    def run(self):
        print ("Start Process： " + self.name)
        gen_imgs(self.name, self.dataIds, self.sample_xywh, self.img_fns)
        print ("End Process：" + self.name)


if __name__=='__main__':

    tianchiCom = '/home/admin/jupyter/Data/'
    pos_path = tianchiCom + 'train/'
    label_path = pos_path
    marksave_dir='/home/admin/jupyter/zxy/'
    img_path = 'sample_pos_data_1024/'
    img_dir = marksave_dir+img_path
    if not(os.path.exists(marksave_dir)):
        os.mkdir(marksave_dir)
    if not(os.path.exists(img_dir)):
        os.mkdir(img_dir)
    W=1024
    H=1024
    for r,d,fs in os.walk(pos_path):
        break
    #pos id
    #dataIds = []
    #for f in fs:
    #    if f[-1] == 'n':
    #        dataIds.append(f[:-5])
    dataIds = ['2393','8484']

    categories ={'ASC-H':0,'ASC-US':1,'HSIL':2,'LSIL':3,'Candida':4,'Trichomonas':5}
    weight=[6,6,12,10,20,3]
    
    pbar = tqdm(total = len(dataIds))
    pbar.set_description('Generating annotations...')
    
    sample_xywh, img_fns = gen_csv()
    th = myProcess(1, "Process-"+str(1), dataIds, sample_xywh, img_fns)
    th.start()
    th.join()
    exit()
    cp = [i for i in range(0,len(dataIds),int(len(dataIds)/9))]
    cp[-1] = len(dataIds)
    threads = []
    for i in range(len(cp)-1):
        th = myProcess(i+1, "Process-"+str(i+1), dataIds[cp[i]:cp[i+1]], sample_xywh, img_fns)
        threads.append(th)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()




