from datetime import datetime
import os
import time

while True:
    dt=datetime.now()
    rr = 1
    if int(dt.minute) == 30:
        if not os.popen('fuser -v /dev/nvidia*').readlines():
            os.system('nohup ./tools/dist_train.sh  configs/faster_rcnn_r50_fpn_1x.py 2 --work_dir work_dirs/f_fpn_v0.7 --validate --resume_from work_dirs/f_fpn_v0.7/latest.pth >f_fpn_r50_3.out 2>&1 &')
    if int(dt.hour) == 5 and int(dt.minute) == 00:
        nvidia_pids = os.popen('fuser -v /dev/nvidia*').readlines()[0].split(' ')[1:]
        for pid in nvidia_pids:
            os.system('kill '+str(pid))
        time.sleep(30)
        rr = os.system('nohup ./tools/dist_test.sh configs/faster_rcnn_r50_fpn_1x.py work_dirs/f_fpn_v0.7/latest.pth 2 --out auto_test_res_'+ str(dt.day) +'d.pkl --eval bbox >auto_test_'+ str(dt.day) +'d.out 2>&1 &')
    time.sleep(30)
    if rr == 0: print('Runing test in ',dt.hour,':',dt.minute+1)
    else: print('Nothing have done in ',dt.hour,':',dt.minute)