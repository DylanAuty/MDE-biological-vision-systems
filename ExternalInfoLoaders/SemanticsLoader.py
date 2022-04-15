# SemanticsLoader.py
# Implements SemanticsLoader class
# Owned by train/eval functions, responsible for initial loading into memory of semantic embeddings and
# for performing lookup on the raw semantics spat out by the dataloader as part of the batch each iteration.
# Embeddings accessible through a member variable, embedding function is a member function.

import sys
import numpy as np
import torch
import torch.nn.functional as F


class SemanticsLoader():
	def __init__(self, args):
		self.args = args

		self.embeddings_path = None				# Will store path to required embeddings file.
		self.human_sizes_path = None			# Will store path to the human-sizes file
		
		self.word_embeddings_semantics = None	# Will store torch tensor containing word embeddings for each class.
		self.human_sizes = None					# Will store torch tensor containing human-sizes for the ade20k-places classes.

		# Set paths to external data, according to args.
		self.set_semantics_path()
		self.set_human_sizes_path()

		# Load external data from disk into memory
		self.load_word_embeddings()
		self.load_human_sizes()

	
	def set_semantics_path(self):
		# Sets the path to the semantic embeddings file dependent on the args.
		if self.args.use_semantics is not None:
			# 300d GloVe embeddings
			if self.args.use_semantics == "glove":		
				self.embeddings_path = "data/ade20k_150_classes_glove_840b_300d_embeddings.npy"

			# 25d GloVe embeddings trained on Twitter.
			elif self.args.use_semantics == "glove-25d" or self.args.use_semantics == "glove-25d-inst-areas":
				self.embeddings_path = "data/ade20k_150_classes_glove_twitter_27b_25d_embeddings.npy"

			# Using the 100 ade20k-places classes (from the instance segmentation model).
			elif "ade20k-places" in self.args.use_semantics:
				# Use consistent-but-random embeddings instead of anything from a language model.
				if "random" in self.args.use_semantics:
					self.embeddings_path = "data/ade20k_places_classes_25d_embeddings_random.npy"

				# Using ade20k-places classes (from instance seg model),
				# but using the Twitter-trained glove-25d embeddings
				elif "glove-25d" in self.args.use_semantics:
					if "size_shuffled" in self.args.use_semantics:
						self.embeddings_path = "data/ade20k_places_classes_glove_twitter_27b_25d_embeddings_shuffled.npy"
					else:
						self.embeddings_path = "data/ade20k_places_classes_glove_twitter_27b_25d_embeddings.npy"


	def set_human_sizes_path(self):
		# Sets the path to the human-sizes file dependent on args.
		# This is for use with ade20k-places only - the size file is 101 classes.
		# (The file is just poorly named...)
		if self.args.use_semantics is not None:
			if "human-sizes" in self.args.use_semantics:
				if "ade20k-places" in self.args.use_semantics:
					# human-sizes can only be used with the ade20k-places classes at the moment.
					if "shuffled" in self.args.use_semantics:
						self.human_sizes_path = "data/ade20k_classes_abs_sizes_shuffled.npy"
					else:
						self.human_sizes_path = "data/ade20k_classes_abs_sizes.npy"
				else:
					sys.exit("Error: human-sizes not implemented for semantics other than ade20k-places.")


	def load_word_embeddings(self):
		# Load word embeddings from disk.
		if self.embeddings_path is not None:
			self.word_embeddings_semantics = np.load(self.embeddings_path)
			self.word_embeddings_semantics = torch.from_numpy(self.word_embeddings_semantics)

	
	def load_human_sizes(self):
		# Load absolute sizes for the ade20k-places classes from disk.
		if self.human_sizes_path is not None:
			self.human_sizes = np.load(self.human_sizes_path)
			self.human_sizes = torch.from_numpy(self.human_sizes)

	
	def get_semantics_inst_areas(self, semantics_raw):
		# Handles the glove-25d-inst-areas case, which needs per-class areas (where class labels come from the instance seg model)
		# Working out areas per class in the image, normalised to total image area.
		# Needs vectorisation
		semantics_area = torch.DoubleTensor(semantics_raw.shape).cuda()
		total_area = semantics_raw.shape[2] * semantics_raw.shape[3]
		for label in torch.unique(semantics_raw):
			for im in range(0, semantics_raw.shape[0]):
				tmp_indices = (semantics_raw[im] == label).nonzero(as_tuple=True)	# Get locations where class is label
				label_area = tmp_indices[0].shape[0] / total_area	# Count number of pixels with this class
				semantics_area[im, tmp_indices[0], tmp_indices[1], tmp_indices[2]] = label_area
		return semantics_area


	def get_semantics(self, batch):
		# Callable with a batch that may or may not contain semantics.
		# If there's no semantics in the batch, will return None, None.
		# If only raw semantics are required, then returns semantics_raw, semantics_raw.
		# semantics may contain other stuff stacked on top

		semantics_raw = None
		semantics = None

		if self.args.use_semantics is not None:
			semantics_raw = batch['semantics']

			# First, preprocess the raw semantics (if needed)
			if "ade20k-places" in self.args.use_semantics:
				# 100 classes, plus 1 background class. Classes 0-99 are the 100 real classes. Class 100 (the 101st) is bg.
				semantics_raw[semantics_raw > 100] = 100	# If invalid class is detected, call it background.
				semantics_raw[semantics_raw < 0] = 100	# If invalid class is detected, call it background.

			# Sort out requests for raw semantics
			if "raw" in self.args.use_semantics:
				semantics = semantics_raw.float().cuda()
			# Do word embedding for non-raw semantics
			else:
				semantics = self.word_embeddings_semantics.index_select(0, semantics_raw.view(-1))
				semantics = semantics.view(semantics_raw.shape[0], semantics_raw.shape[1], semantics_raw.shape[2], semantics_raw.shape[3], self.word_embeddings_semantics.shape[1])
				semantics = semantics.squeeze(1).permute(0, 3, 1, 2).contiguous()
				if "ade20k-places" in self.args.use_semantics:
					semantics = semantics.float()
				semantics = semantics.cuda()

			# If there's anything to add to the semantics, do that now.
			# Per-class sizes
			if "inst-areas" in self.args.use_semantics:
				semantics_area = self.get_semantics_inst_areas(semantics_raw)
				semantics = torch.cat((semantics, semantics_area), dim=1)

			# Human sizes (approx. average absolute dimensions for each class in ade20k-places)
			if self.human_sizes is not None:
				instance_abs_sizes = self.human_sizes.index_select(0, semantics_raw.view(-1))
				instance_abs_sizes = instance_abs_sizes.view(semantics_raw.shape[0], semantics_raw.shape[1], semantics_raw.shape[2], semantics_raw.shape[3], 3)
				instance_abs_sizes = instance_abs_sizes.squeeze(1).permute(0, 3, 1, 2).contiguous().float().cuda()
				semantics = torch.cat((semantics, instance_abs_sizes), dim=1)

		return semantics_raw, semantics

