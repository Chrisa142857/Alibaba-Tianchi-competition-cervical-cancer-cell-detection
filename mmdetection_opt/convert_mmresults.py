import json
from tqdm import tqdm
import pickle
import os
######################################## COCO ##########################################
res_path = '/home/admin/jupyter/wei/mmdetection-master/auto_test_res_16d.pkl.bbox.json'
ann_path = '/home/admin/jupyter/wei/mmdetection-master/data/coco/annotations/instances_test2017_11cls.json'
save_path = '/home/admin/jupyter/wei/mmdetection-master/res_for_submit/f_fpn_ep7_v0.6/'
#categories = ['ASC-H1','ASC-H3','ASC-US1','ASC-US3','HSIL1','HSIL3','LSIL1','LSIL3','Candida1','Candida3','Trichomonas1']

categories = ['ASC-H','ASC-US','HSIL','LSIL','Candida','Trichomonas']

if not(os.path.exists(save_path)):
    os.mkdir(save_path)

with open(res_path) as f:
    res = json.load(f)
with open(ann_path) as f:
    ann = json.load(f)
json_saved = {}

pbar = tqdm(total = len(res))
pbar.set_description('Converting...')

for item in res:
    fn = ann['images'][item['image_id']]['file_name'][:-4]
    first_id = fn.find('_')
    second_id = fn[first_id+1:].find('_') + first_id + 1
    third_id = fn[second_id+1:].find('_') + second_id + 1
    forth_id = fn[third_id+1:].find('_') + third_id + 1
    dataId = fn[:first_id]
    tlx = int(fn[first_id+1:second_id])
    tly = int(fn[second_id+1:third_id])
    iw = int(fn[third_id+1:forth_id])
    ih = int(fn[forth_id+1:])
    scale_x = iw / 1024
    scale_y = ih / 1024
    if dataId not in json_saved: json_saved[dataId] = []
    if item['score'] < 0.05: continue
    json_saved[dataId].append({
        'x':item['bbox'][0]*scale_x + tlx,
        'y':item['bbox'][1]*scale_y + tly,
        'w':item['bbox'][2]*scale_x,
        'h':item['bbox'][3]*scale_y,
        'p':item['score'],
        'class': categories[item['category_id']-1][:-1]
    })
    pbar.update(1)

for file in json_saved:

    json_name = save_path + file + '.json'
    with open(json_name,'w') as jsonF:
        json.dump(json_saved[file], jsonF, ensure_ascii=False)
        print(str(len(json_saved[file]))+' boxes in '+file)

exit(0)