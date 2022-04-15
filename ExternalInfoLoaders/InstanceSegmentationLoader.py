# InstanceSegmentationLoader.py
# Implements InstanceSegmentationLoader, which handles the loading of instance segmentation information.
# Each train/test/validate function owns one of these.
# Responsible also for loading embeddings into memory, which are accessible through a member variable

import sys
import numpy as np
import torch


class InstanceSegmentationLoader():
	def __init__(self, args):
		self.args = args
	
		self.embeddings_path = None		# Will store path to the word embeddings that will be used with the class labels
		self.human_sizes_path = None	# Will store path to the human-provided rough dimensions for each class.
		
		self.word_embeddings_semantics = None	# Will store word embeddings to be used with the semantic labels
		self.background_class_num = None		# Each dataset uses a different number for the "background" label; this stores it.
		self.human_sizes = None					# Stores the per-class human-sizes

		# Set up paths
		self.set_embeddings_path()
		self.set_human_sizes_path()

		# Load external information (maybe, depending on args)
		self.load_word_embeddings()
		self.load_human_sizes()


	def set_embeddings_path(self):
		# Set the path to the word embeddings to use, depending on the arguments.
		if self.args.use_instance_segmentation is not None:
			# Exit if "raw" is requested. It looks like this may have never worked...?
			if self.args.use_instance_segmentation == "raw":
				sys.exit("Error: raw instance semantics not implemented")


			# Load embeddings to work with MSCOCO instance segmentation model (Mask-RCNN)
			# Embeddings are GloVe-25d, trained on Twitter.
			if self.args.use_instance_segmentation == "coco":
				self.embeddings_path = "data/coco_81_classes_maskrcnn_ordering_glove_twitter_27b_25d_embeddings.npy"
				self.background_class_num = 0

			# Load embeddings for use with the SWIN instance segmentation model
			# Trained on ADE20K-places
			# Embeddings are GloVe-25d Twitter
			elif "ade20k_swin" in self.args.use_instance_segmentation:
				self.embeddings_path = "data/ade20k_places_classes_glove_twitter_27b_25d_embeddings.npy"
				self.background_class_num = 100

			# If adding more datasets, they should be arranged to have no. labels first, then no. channels.
			# i.e. shape should be [N, C]. This is to work with the clamping of each example.


			# Error checking: if lots more datasets are added, this makes sure everything is actually being
			# set by any new blocks of code.
			assert self.embeddings_path is not None
			assert self.background_class_num is not None
	

	def set_human_sizes_path(self):
		# Set path to average absolute per-class human sizes, only works for ade20k-places classes
		# ade20k-places classes are from ade20k_swin model (instance segmentation model)
		if self.args.use_instance_segmentation is not None:
			if "ade20k_swin" in self.args.use_instance_segmentation:
				if "human_sizes" in self.args.use_instance_segmentation:
					if "shuffled" in self.args.use_instance_segmentation:
						self.human_sizes_path = "data/ade20k_classes_abs_sizes_shuffled.npy"
					else:
						self.human_sizes_path = "data/ade20k_classes_abs_sizes.npy"

	
	def load_word_embeddings(self):
		# Load word embeddings from disk into memory
		if self.embeddings_path is not None:
			self.word_embeddings_semantics = np.load(self.embeddings_path)
			self.word_embeddings_semantics = torch.from_numpy(self.word_embeddings_semantics)
	

	def load_human_sizes(self):
		# Load human-provided per-class average dimensions/sizes, for use with ade20k-places classes spat out by
		# the SWIN instance segmentation model.
		if self.human_sizes_path is not None:
			self.human_sizes = np.load(self.human_sizes_path)
			self.human_sizes = torch.from_numpy(self.human_sizes)

	
	def get_instance_segmentation(self, batch):
		# Method called in each iteration. 
		# Returns instance_labels_raw, instance_labels, and instance_areas.

		instance_labels_raw = None
		instance_labels = None
		instance_areas = None

		if self.word_embeddings_semantics is not None:
			instance_labels_raw = batch['instance_labels']
			instance_areas_raw = batch['instance_areas']

			# Error correction of instance_labels_raw
			# Everything outside of the normal range is assigned the background class.
			instance_labels_raw[instance_labels_raw < 0] = self.background_class_num
			instance_labels_raw[instance_labels_raw > (self.word_embeddings_semantics.shape[0] - 1)] = self.background_class_num

			# Do the embedding of the instance_labels_raw using the loaded word semantics
			instance_labels = self.word_embeddings_semantics.index_select(0, instance_labels_raw.view(-1))
			instance_labels = instance_labels.view(instance_labels_raw.shape[0], instance_labels_raw.shape[1], instance_labels_raw.shape[2], instance_labels_raw.shape[3], self.word_embeddings_semantics.shape[1])
			instance_labels = instance_labels.squeeze(1).permute(0, 3, 1, 2).contiguous().cuda()

			# instance_areas are just placed on the device.
			instance_areas = instance_areas_raw.float().cuda()

			# If we're using human-provided sizes, then work them out and tack them on the end of instance_areas
			if self.human_sizes is not None:
				instance_abs_sizes = self.human_sizes.index_select(0, instance_labels_raw.view(-1))
				instance_abs_sizes = instance_abs_sizes.view(instance_labels_raw.shape[0], instance_labels_raw.shape[1], instance_labels_raw.shape[2], instance_labels_raw.shape[3], 3)
				instance_abs_sizes = instance_abs_sizes.squeeze(1).permute(0, 3, 1, 2).contiguous().float().cuda()
				instance_areas = torch.cat((instance_areas, instance_abs_sizes), dim=1)

		return instance_labels_raw, instance_labels, instance_areas



		return instance_labels, instance_areas
