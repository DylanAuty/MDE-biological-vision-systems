#!/bin/bash

tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k.py 1 --cfg-options model.pretrained=checkpoints/swin_base_patch4_window12_384_22kto1k.pth
