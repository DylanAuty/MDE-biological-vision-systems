# glove_load_test.py
# Script to work out how to load the specific embeddings needed from a glove embeddings file.
# The class names as stored in object150_info.csv are offset by 1 relative to the stored predictions.
# i.e. class "0" in the stored predictions is actually class 1 in the object150_info.csv file, class "wall".
# This can be verified in test.py in CSAILVision/semantic-segmentation-pytorch, function visualize_result.
# Generated np array will have dimensions [150, N]. N is the feature vector (embedding) for each of the 150 classes in ADE20K.

import numpy as np
import os

#embeddings_path = os.path.expanduser('~/Documents/GloVe/glove.twitter.27B.25d.txt')	# Twitter 
#embeddings_path = os.path.expanduser('~/Documents/GloVe/glove.840B.300d.txt')
#class_info_path = os.path.expanduser('~/Documents/AdaBins/data/object150_info_swapped_order.csv')

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# Indices of classes come from github.com/matterport/Mask_RCNN, their Dataloader 
# gives different indices and retrieving them requires loading all of COCO.
# This array is the product of doing that, to save the work.
coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

# These are the ade20k Places Challenge classes - 100 classes only, for use with an instance segmentation model.
ade20k_places_class_names = ['bed', 'windowpane', 'cabinet', 'person', 'door', 'table', 'curtain', 'chair', 'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair', 'seat', 'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sink', 'fireplace', 'refrigerator', 'stairs', 'case', 'pool table', 'pillow', 'screen door', 'bookcase', 'coffee table', 'toilet', 'flower', 'book', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'apparel', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship', 'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag', 'mini bike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'trashcan', 'fan', 'plate', 'monitor', 'bulletin board', 'radiator', 'glass', 'clock', 'flag', 'background']	# The last class, "background", is manually added. Annotation numbers "-1" in ade20k_swin outputs correspond to the "background" class. The class "trashcan" was originally "ashcan". "mini bike" was originally "minibike".

# This array is the same as the previous, but "BG" has been changed to "background"
coco_class_names_modified = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

first = True
found = False

# Change output according to size of embedding being used and number of classes in dataset
#output = np.zeros([101, 25])
output = np.zeros([101, 128])
#output = np.zeros([81, 300])
idx = 0
failed = []

"""
# A loop for getting embeddings of ADE20K class names
with open(class_info_path, 'r') as info_f:
	for class_info_line in info_f:
		if first:
			first = False
			continue
		else:
			class_name = class_info_line.rstrip().split(',')[5].split(';', 1)[0]
		print(f"Searching for {class_name}")
		with open(embeddings_path, 'r') as embeddings_f:
			for line in embeddings_f:
				split = line.split(" ", 1)
				if split[0] == class_name:
					print(f"Found {class_name} : {idx}")
					found = True
					output[idx] = np.fromstring(split[1], dtype=float, sep=' ')
					idx += 1
					break
		if found == False:
			failed.append(class_name)

		found = False
"""

# A loop for getting embeddings for the coco class names or ade20k, ordered as they are in 
# github.com/matterport/Mask_RCNN after loading COCO using their dataloader or as in the 
# class info .txt file in the placeschallenge repo for COCO or ADE20K respectively.
#for class_name in coco_class_names_modified:
for class_name in ade20k_places_class_names:
	# Check if the class name is multi-word - if it is, find both words and add embeddings
	# If either fails, count whole class name as a failure.
	components = class_name.split()
	found = True
	for component in components:
		found_component = False
		print(f"Searching for {class_name} ({component})")
		with open(embeddings_path, 'r') as embeddings_f:
			for line in embeddings_f:
				split = line.split(" ", 1)
				if split[0] == component:
					output[idx] += np.fromstring(split[1], dtype=float, sep=' ')
					found_component = True
					break
			if found_component is False:
				print(f"No match for {component}")
				found = False
				break

	idx += 1
	if found == False:
		failed.append(class_name)
	

#np.save('./ade20k_places_classes_glove_twitter_27b_25d_embeddings.npy', output)
print("Done.")
print("Failed:")
print(failed)

