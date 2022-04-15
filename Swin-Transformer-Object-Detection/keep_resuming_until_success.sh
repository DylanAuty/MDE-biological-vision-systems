#!/bin/bash

tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k.py 1 --cfg-options resume_from=./work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k/latest.pth
while [ $? -ne 0 ]; do
	echo "Trying again..." | tee -a ./failure_log.txt
	sleep 5;
	pkill -f python
	sleep 5;

	tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k.py 1 --cfg-options resume_from=./work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k/latest.pth
done
