# nyud2_inference.py 
# Runs inference on all files in a directory.

import argparse
import sys
import os
import mmcv
from mmcv import Config, DictAction
from mmdet.apis import init_detector, inference_detector
from mmdet.models import build_detector
import torch
import fnmatch
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle


def find_recursive(root_dir, ext='.jpg'):
	files = []
	for root, dirnames, filenames in os.walk(root_dir):
		for filename in fnmatch.filter(filenames, '*' + ext):
			files.append(os.path.join(root, filename))
	return files


def check_prediction_match(image_path, collapsed_masks, areas_masks, name):
	"""Checks provided collapsed masks and area masks with the files already saved to disk.
	"""
	collapsed_loaded = np.load(image_path.replace('rgb', 'instance_labels_' + name).replace('.jpg', '.npz'), allow_pickle=True) 
	areas_loaded = np.load(image_path.replace('rgb', 'instance_areas_' + name).replace('.jpg', '.npz'), allow_pickle=True) 
	collapsed_loaded = collapsed_loaded['arr_0']
	areas_loaded = areas_loaded['arr_0']
	if (collapsed_masks != collapsed_loaded).any() or (areas_loaded != areas_masks).any():
		return False
	else:
		return True


def load_depth_map(image_path):
	"""Loads in the saved depth map for a given image as a numpy array
	"""
	depth_gt = Image.open(image_path.replace('rgb', 'sync_depth').replace('.jpg', '.png')) 
	depth_gt = np.asarray(depth_gt, dtype=np.float32)
	depth_gt = np.expand_dims(depth_gt, axis=2)
	depth_gt = depth_gt / 1000.0

	return depth_gt


def save_prediction(image_path, collapsed_masks, area_masks, name):
	np.savez_compressed(image_path.replace('rgb', 'instance_labels_' + name).replace('.jpg', '.npz'), collapsed_masks) 
	np.savez_compressed(image_path.replace('rgb', 'instance_areas_' + name).replace('.jpg', '.npz'), area_masks) 


def remove_leading_slash(s):
	if s[0] == '/' or s[0] == '\\':
		return s[1:]
	return s


def main(args):
	config_file = args.config
	checkpoint_file = args.checkpoint

	# Build model
	model = init_detector(config_file, checkpoint_file, device='cuda:0')

	# Grab all the image filepaths
	if os.path.isdir(args.images):
		files = find_recursive(args.images)
	elif args.images.lower().endswith('txt'):
		sys.exit("Make sure to edit base_dir on line 77 if you want to use this. It should align with the file passed in.")
		# Indicates a txt file where each line is a path
		base_dir = "data/nyu/official_splits/test"
		files = []
		with open(args.images, 'r') as f:
			filenames = f.readlines()
		for line in filenames:
			files.append(os.path.join(base_dir, remove_leading_slash(line.split()[0])))
	else:
		files = [args.images]
	
	# Set up array for histogram of classes
	# We make a list of lists: One list for each class, each being a list of (area, ave_depth, depth_std_deviation)
	# for a given instance
	area_depth_std_points = [ [] for _ in range(101)]	# 100 classes, then all -1s in the collapsed_masks is class 101 (background)

	# An array to store the list of failures
	failures = []
	
	for img in tqdm(files):
		result = inference_detector(model, img)		# Result is an array of tuples. Each tuple is (bbox, segm).
		# For each of these, there are N entries in a list, where each entry corresponds to one of the N classes in the dataset.
		# For each class, there is one mask per instance of that class.

		depth_gt = load_depth_map(img).squeeze()	# 480x680, depth map for this image.

		area_masks = None
		collapsed_masks = None
		collapsed_areas = None

		# Go through all class masks. Make them into areas per instance, and also class labels instead of booleans.
		for class_id in range(0, len(result[1])):
			segm_mask = result[1][class_id]
			bbox = result[0][class_id]	# bboxes are [x1, y1, x2, y2, score]
			for i in range(0, len(segm_mask)):	# Go through each instance of this class in this image
				if collapsed_masks is None:
					collapsed_masks = np.ones(segm_mask[i].shape, dtype=np.int32) * -1	# Merging all ROI classes into one layer
				if collapsed_areas is None:
					collapsed_areas = np.zeros(segm_mask[i].shape, dtype=np.int32)	# Produce per-pixel, per-roi area map.

				## Line 117 should be used for computing bbox area, and line 118 for mask area.
				#area = (bbox[i][2] - bbox[i][0]) * (bbox[i][3] - bbox[i][0])
				area = segm_mask[i].sum()	# Instance areas are normalised inside the AdaBins model forward()
				#area_mask = np.expand_dims(np.where(segm_mask[i] == True, area, 0), axis=-1)
				depth_mean = depth_gt[segm_mask[i]].mean()
				depth_std = depth_gt[segm_mask[i]].std()
				
				area_depth_std_points[class_id].append([area, depth_mean, depth_std])

				# The collapsed_masks/areas contains overlapping information about all classes.
				np.putmask(collapsed_masks, segm_mask[i] == True, class_id)
				np.putmask(collapsed_areas, segm_mask[i] == True, area)

		# Once all classes and their instances in the image have been processed, we save the result.
		# If computing bounding box areas, change name to "ade20k_swin_bbox".
		save_prediction(img, collapsed_masks, collapsed_areas, "ade20k_swin")

		# Confirm predictions match those on file
		#if check_prediction_match(img, collapsed_masks, collapsed_areas, "ade20k_swin") == False:
		#	failures.append(img)

	#with open('./area_depth_std_points_ade20k_places_train_bboxes.pkl', 'wb') as outfile:
	#	pickle.dump(area_depth_std_points, outfile)
	
	#with open('./failure_log_1.txt', mode='a') as outfile2:
	#	for thing in failures:
	#		outfile2.write(thing + '\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, help="Path to config file.")
	parser.add_argument('--checkpoint', type=str, help="Path to checkpoint file.")
	parser.add_argument('--images', type=str, help="Path to image file or directory. If a directory, will run inference on all in directory. If a file then will run inference on just the file.")
	
	args = parser.parse_args()

	main(args)
