import json
import os 

root_dir = '/home/admin/jupyter/wei/mmdetection-master/data/coco/'
ann_dir = root_dir + 'annotations/'
old_json = json.load(open(ann_dir+'instances_test2017_6cls.json','r'))
info = old_json['info']
license = old_json['license']

#categories = [{'id': 1,'name': 'ASC-H'},{'id': 2,'name': 'ASC-US'},{'id': 3,'name': 'HSIL'},{'id': 4,'name': 'LSIL'},{'id': 5,'name': 'Candida'},{'id': 6,'name': 'Trichomonas'}]
#categories = [{'id': 1,'name': 'ASC-H1'},{'id': 2,'name': 'ASC-H3'},{'id': 3,'name': 'ASC-US1'},{'id': 4,'name': 'ASC-US3'},{'id': 5,'name': 'HSIL1'},{'id': 6,'name': 'HSIL3'},{'id': 7,'name': 'LSIL1'},{'id': 8,'name': 'LSIL3'},{'id': 9,'name': 'Candida1'},{'id': 10,'name': 'Candida3'},{'id': 11,'name': 'Trichomonas1'}]
categories = [{'id': 1,'name': 'ASC-H1'},{'id': 2,'name': 'ASC-H2'},{'id': 3,'name': 'ASC-H3'},{'id': 4,'name': 'ASC-US1'},{'id': 5,'name': 'ASC-US2'},{'id': 6,'name': 'ASC-US3'},{'id': 7,'name': 'HSIL1'},{'id': 8,'name': 'HSIL2'},{'id': 9,'name': 'HSIL3'},{'id': 10,'name': 'LSIL1'},{'id': 11,'name': 'LSIL2'},{'id': 12,'name': 'LSIL3'},{'id': 13,'name': 'Candida1'},{'id': 14,'name': 'Candida2'},{'id': 15,'name': 'Candida3'},{'id': 16,'name': 'Trichomonas1'}]

data_test = {}
data_test['info'] = info
data_test['license'] = license
data_test['annotations'] = []
data_test['images'] = old_json['images']
data_test['categories'] = categories
with open(ann_dir + 'instances_test2017_16cls.json','w') as f:
    json.dump(data_test, f)
