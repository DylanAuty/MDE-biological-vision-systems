# Handles the .pkl file containing data about the size and average distance of every instance in 
# the ADE20K places dataset.

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats

# ade20k classes as used in the Places challenge. Class 100 (the 101st) has been set to be background.
ade20k_classes = ['bed', 'windowpane', 'cabinet', 'person', 'door', 
		'table', 'curtain', 'chair', 'car', 'painting', 
		'sofa', 'shelf', 'mirror', 'armchair', 'seat', 
		'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 
		'railing', 'cushion', 'box', 'column', 'signboard',
		'chest of drawers', 'counter', 'sink', 'fireplace', 'refrigerator',
		'stairs', 'case', 'pool table', 'pillow', 'screen door',
		'bookcase', 'coffee table', 'toilet', 'flower', 'book',
		'bench', 'countertop', 'stove', 'palm', 'kitchen island',
		'computer', 'swivel chair', 'boat', 'arcade machine', 'bus',
		'towel', 'light', 'truck', 'chandelier', 'awning',
		'streetlight', 'booth', 'television receiver', 'airplane', 'apparel',
		'pole', 'bannister', 'ottoman', 'bottle', 'van',
		'ship', 'fountain', 'washer', 'plaything', 'stool',
		'barrel', 'basket', 'bag', 'minibike', 'oven',
		'ball', 'food', 'step', 'trade name', 'microwave',
		'pot', 'animal', 'bicycle', 'dishwasher', 'screen',
		'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
		'tray', 'ashcan', 'fan', 'plate', 'monitor', 
		'bulletin board', 'radiator', 'glass', 'clock', 'flag', 
		'background']

# Structure of pkl is one list for each of the 100 classes (class 101 is background).
# Each of these lists is a list of each instance's [area, depth_mean, depth_std_deviation]
with open('data/area_depth_std_points_ade20k_places_train_bboxes.pkl', 'rb') as f:
	data = pickle.load(f)

save_folder = './ade20k_places_scatterplots_bbox'
for i in range(0, len(data)):
	areas = np.array([])
	depth_means = np.array([])
	for inst in data[i]:
		if not (np.isinf(inst[0]) or np.isinf(inst[1]) or np.isnan(inst[0]) or np.isnan(inst[1])):
			tmp_area = inst[0] / (307200.0)
			if tmp_area > 0.00:
				areas = np.append(areas, tmp_area)	# 640 * 480 = 307200
				depth_means = np.append(depth_means, inst[1])
	if len(areas) < 2:
		continue
	corr, p_values = scipy.stats.pearsonr(areas, depth_means)
	m, b = np.polyfit(areas, depth_means, 1)
	
	title = f"Class {i}: {ade20k_classes[i]}, r={corr:.3f}"
	plt.plot(areas, depth_means, ',')
	#plt.plot(areas, m * areas + b)
	plt.title(title)
	plt.xlabel('Area (proportion of frame)')
	plt.ylabel('Mean depth (m)')
	plt.xlim(xmin=0)
	plt.ylim(ymin=0)
	plt.savefig(os.path.join(save_folder, f"{i}_{ade20k_classes[i]}.png"), dpi=150)
	print(f"{i}: {ade20k_classes[i]}")
	plt.clf()

