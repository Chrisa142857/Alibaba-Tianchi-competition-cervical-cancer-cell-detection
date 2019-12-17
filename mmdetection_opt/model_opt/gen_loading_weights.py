import torch
#num_classes = 7
model_coco = torch.load("resnet50-19c8e357.pth")
model_old = torch.load("../work_dirs/f_fpn_v0.6/latest.pth")

#for key in model_old['state_dict']:
#    part_n = key[:key.find('.')]
#    if part_n == 'backbone' and (key[key.find('.')+1:] in list(model_coco.keys())):
#        print(key+': ',model_old['state_dict'][key].size(),'->',model_coco[key[key.find('.')+1:]].size())
#        model_old['state_dict'][key] = model_coco[key[key.find('.')+1:]]

#del model_81['state_dict']['rpn_head.rpn_cls.weight']
#del model_81['state_dict']['rpn_head.rpn_reg.weight']
#del model_81['state_dict']['rpn_head.rpn_cls.bias']
#del model_81['state_dict']['rpn_head.rpn_reg.bias']

need_ids = [0,1,3,5,7,9,11]
reg_needs = []
for id in need_ids:
    reg_needs.extend([id*4+i for i in range(4)])

# weight
model_old["state_dict"]["bbox_head.fc_cls.weight"] = model_old["state_dict"]["bbox_head.fc_cls.weight"][need_ids, :]
model_old["state_dict"]["bbox_head.fc_reg.weight"] = model_old["state_dict"]["bbox_head.fc_reg.weight"][reg_needs, :]
# bias
model_old["state_dict"]["bbox_head.fc_cls.bias"] = model_old["state_dict"]["bbox_head.fc_cls.bias"][need_ids]
model_old["state_dict"]["bbox_head.fc_reg.bias"] = model_old["state_dict"]["bbox_head.fc_reg.bias"][reg_needs]
for key in model_old['state_dict']:
    print(key+': ',model_old['state_dict'][key].size())

#save new model
torch.save(model_old,"v0.6_epoch8_6cls.pth")
