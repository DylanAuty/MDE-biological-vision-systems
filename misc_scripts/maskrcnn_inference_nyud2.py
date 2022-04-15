# inference_nyud2.py
# Run Mask-RCNN inference on NYUD2, using specified checkpoint.
# Aim is to save segmentation information for all examples in the provided directory.
# Code based on Mask_RCNN/samples/demo.ipynb

"""
Place this script into the Mask-RCNN directory and change the path to the dataset.

Some modification to Mask-RCNN is necessary to make this work with Tensorflow > 2 and python > 3.6.
This was tested with python3.8 and tensorflow 2.4.1.
You could also work with the dependencies the repo was written for, if desired.

This comment/issue detailed the basic process:
https://github.com/matterport/Mask_RCNN/issues/1070#issuecomment-740430758

1. Upgrade the Mask-RCNN code using Tensorflow's upgrade command, which comes installed with Tensorboard>2
	```
	cd /path/to/Mask_RCNN`
	tf_upgrade_v2 --intree ./ --inplace --reportfile report.txt
	```
	This ports some of the outdated Tensorflow 1.xx commands to their Tensorflow 2.xx equivalents

2. Replace the following line of mrcnn/model.py (on line 951):
	```
	mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
	```
	with this if-else block:
	```
    if s[1]==None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
	```

3. Change this line of mrcnn/model.py (somewhere around line 702-ish):
	```
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	```
	with this line
	```
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
	```

4. In the try/except on or around line 2109, change both keras imports:
	```
	from keras.engine import saving
	from keras.engine import topology as saving
	```
	replace these with
	```
	from tensorflow.python.keras.engine import saving
	from tensorflow.python.keras.engine import topology as saving
	```

5. Replace the saving function calls on or around line 2133, in BOTH the if and else blocks:
	```saving.<FUNCTION>```
	Replace with
	```saving.hdf5_format.<FUNCTION>```
	where `<FUNCTION>` is one of the functions in the `saving` library.
"""

import os
import sys
import fnmatch
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import samples.coco.coco as coco
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#%matplotlib inline 


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


def save_prediction(image_path, collapsed_masks, area_masks, name):
	np.save(image_path.replace('rgb', 'instance_labels_' + name).replace('.jpg', '.npy'), collapsed_masks) 
	np.save(image_path.replace('rgb', 'instance_areas_' + name).replace('.jpg', '.npy'), collapsed_areas) 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Define paths and names
in_imgs = os.path.expanduser('~/WorkingDatasets/nyu')		# Run inference on all images beneath this path
result_dir = os.path.expanduser('~/WorkingDatasets/nyu')	# Save images to this path, using same file structure as input
name = 'coco'									# Used in prediction file names


class InferenceConfig(coco.CocoConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()

# Loading model
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

"""
# Handling class names - Dataset class overwrites original class IDs.
# Getting this would normally require loading the dataset then getting the dataset.class_names property
# Instead the COCO class names are below, in class_names
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()
print(dataset.class_names)
"""

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Running Object Detection

# generate testing image list
if os.path.isdir(in_imgs):
	imgs = find_recursive(in_imgs)
else:
	imgs = [in_imgs]
assert len(imgs), "imgs should be a path to image (.jpg) or directory."

if not os.path.isdir(result_dir):
	os.makedirs(result_dir)

"""
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
"""
imcounter = 0

for image_path in imgs:
	print(f"Image: {imcounter}")
	image = skimage.io.imread(image_path)

	# Run detection
	results = model.detect([image], verbose=0)

	# Visualize results
	r = results[0]

	# Get areas for each region of interest (there are N rois detected in the image)
	masks = np.asarray(results[0]['masks'])	# HxWxN
	areas = masks.sum(axis=(0, 1))			# N, area in pixels for each roi

	# Lazy for loop implementation because this needs to be run only once
	collapsed_masks = np.zeros((masks.shape[0], masks.shape[1]))	# Merging all ROI classes into one layer
	collapsed_areas = np.zeros((masks.shape[0], masks.shape[1]))	# Produce per-pixel, per-roi area map.
	for i in range(0, masks.shape[2]):
		np.putmask(collapsed_masks, masks[:, :, i], i)
		np.putmask(collapsed_areas, masks[:, :, i], areas[i])

	save_prediction(image_path, collapsed_masks, collapsed_areas, name)
	imcounter += 1


"""
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
"""
