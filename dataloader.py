# This dataloader is a modified version of the original AdaBins one, which itself is mostly derived from the BTS implementation.

import os
import sys
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _is_pil_image(img):
	return isinstance(img, Image.Image)


def _is_numpy_image(img):
	return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
	return transforms.Compose([
		ToTensor(mode=mode)
	])


class DepthDataLoader(object):
	def __init__(self, args, mode):
		if mode == 'train':
			self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
			if args.distributed:
				self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
			else:
				self.train_sampler = None

			self.data = DataLoader(self.training_samples, args.batch_size,
								   shuffle=(self.train_sampler is None),
								   num_workers=args.num_threads,
								   pin_memory=True,
								   sampler=self.train_sampler)

		elif mode == 'online_eval':
			self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
			if args.distributed:  # redundant. here only for readability and to be more explicit
				# Give whole test set to all processes (and perform/report evaluation only on one) regardless
				self.eval_sampler = None
			else:
				self.eval_sampler = None
			self.data = DataLoader(self.testing_samples, 1,
								   shuffle=False,
								   num_workers=1,
								   pin_memory=False,
								   sampler=self.eval_sampler)

		elif mode == 'test':
			self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
			self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

		else:
			print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
	if s[0] == '/' or s[0] == '\\':
		return s[1:]
	return s


class DataLoadPreprocess(Dataset):
	def __init__(self, args, mode, transform=None, is_for_online_eval=False):
		self.args = args
		if mode == 'online_eval':
			with open(args.filenames_file_eval, 'r') as f:
				self.filenames = f.readlines()
		else:
			with open(args.filenames_file, 'r') as f:
				self.filenames = f.readlines()

		self.mode = mode
		self.transform = transform
		self.to_tensor = ToTensor
		self.is_for_online_eval = is_for_online_eval

	def __getitem__(self, idx):
		sample_path = self.filenames[idx]
		focal = float(sample_path.split()[2])

		if self.mode == 'train':
			if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
				image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
				depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
			else:
				image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[0]))
				depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[1]))
				
				if self.args.use_semantics is not None:
					if "ade20k-places" not in self.args.use_semantics:
						semantics_raw_path = image_path.replace('rgb', 'semantic_seg').replace('.jpg', '.npy')
					else:
						semantics_raw_path = image_path.replace('rgb', 'instance_labels_' + "ade20k_swin").replace('.jpg', '.npz')

				if self.args.use_instance_segmentation is not None:
					if "ade20k_swin" in self.args.use_instance_segmentation:
						instance_labels_raw_path = image_path.replace('rgb', 'instance_labels_' + 'ade20k_swin').replace('.jpg', '.npz')
						if "bbox" in self.args.use_instance_segmentation:
							instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + 'ade20k_swin_bbox').replace('.jpg', '.npz')
						else:
							instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + 'ade20k_swin').replace('.jpg', '.npz')
					else:
						instance_labels_raw_path = image_path.replace('rgb', 'instance_labels_' + self.args.use_instance_segmentation).replace('.jpg', '.npy')
						instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + self.args.use_instance_segmentation).replace('.jpg', '.npy')


			image = Image.open(image_path)
			depth_gt = Image.open(depth_path)

			# Loading semantics and instance segmentation if needed. 
			if self.args.use_semantics is not None:
				if "ade20k-places" not in self.args.use_semantics:
					semantics_raw = np.load(semantics_raw_path).astype(np.ubyte)	# Needed for Image conversion later
					semantics = semantics_raw	# Double assignment is a hangover from a different way of organising semantic/instance loading.
				else:
					# allow_pickle=True handles case where there are no predictions in .npz files.
					# If loading ade20k_swin, we're loading npz files instead of npys. They need to be handled differently.
					semantics_raw = np.load(semantics_raw_path, allow_pickle=True)
					semantics_raw = semantics_raw['arr_0']
					# If semantic labels are None, then it means that Swin output no predictions for that image.
					# In this case, replace with default empty arrays (0 for area, -1 for labels)
					if len(semantics_raw.shape) != 2:
						semantics_raw = np.ones((image.size[1], image.size[0]), dtype=np.int32) * -1
					semantics = semantics_raw.astype(np.ubyte)	# Needed for Image conversion

				semantics = Image.fromarray(semantics)	# This is done so the augmentation code doesn't need rewriting


			if self.args.use_instance_segmentation is not None:
				# allow_pickle=True handles case where there are no predictions in .npz files.
				instance_labels_raw = np.load(instance_labels_raw_path, allow_pickle=True)
				instance_areas_raw = np.load(instance_areas_raw_path, allow_pickle=True)
				if "ade20k_swin" in self.args.use_instance_segmentation:
					# If loading ade20k_swin, we're loading npz files instead of npys. They need to be handled differently.
					instance_labels_raw = instance_labels_raw['arr_0']
					instance_areas_raw = instance_areas_raw['arr_0']
					
					# If either of these are None, then Swin output no predictions for that image.
					# In this case, replace with default empty arrays (0 for area, -1 for labels)
					if len(instance_labels_raw.shape) != 2:
						instance_labels_raw = np.ones((image.size[1], image.size[0]), dtype=np.int32) * -1
					if len(instance_areas_raw.shape) != 2:
						instance_areas_raw = np.zeros((image.size[1], image.size[0]), dtype=np.int32)

				instance_labels = instance_labels_raw
				instance_areas = instance_areas_raw

				# Convert both to PIL image, using I mode (32 bit signed integer pixels
				# If the conversions don't work, it's because the labels are empty, in which case
				# they get manually replaced with the empty ones.
				instance_labels = Image.fromarray(instance_labels, mode="I")
				instance_areas = Image.fromarray(instance_areas, mode="I")

			if self.args.do_kb_crop is True:
				height = image.height
				width = image.width
				top_margin = int(height - 352)
				left_margin = int((width - 1216) / 2)
				depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
				image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
				if self.args.use_semantics is not None:
					semantics = semantics.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
				if self.args.use_instance_segmentation is not None:
					instance_labels = instance_labels.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
					instance_areas = instance_areas.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

			# To avoid blank boundaries due to pixel registration
			if self.args.dataset == 'nyu':
				depth_gt = depth_gt.crop((43, 45, 608, 472))
				image = image.crop((43, 45, 608, 472))
				if self.args.use_semantics is not None:
					semantics = semantics.crop((43, 45, 608, 472))
				if self.args.use_instance_segmentation is not None:
					instance_labels = instance_labels.crop((43, 45, 608, 472))
					instance_areas = instance_areas.crop((43, 45, 608, 472))

			if self.args.do_random_rotate is True:
				random_angle = (random.random() - 0.5) * 2 * self.args.degree
				image = self.rotate_image(image, random_angle)
				depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
				if self.args.use_semantics is not None:
					semantics = self.rotate_image(semantics, random_angle, flag=Image.NEAREST)
				if self.args.use_instance_segmentation is not None:
					instance_labels = self.rotate_image(instance_labels, random_angle, flag=Image.NEAREST)
					instance_areas = self.rotate_image(instance_areas, random_angle, flag=Image.NEAREST)

			image = np.asarray(image, dtype=np.float32) / 255.0
			depth_gt = np.asarray(depth_gt, dtype=np.float32)
			depth_gt = np.expand_dims(depth_gt, axis=2)
			if self.args.use_semantics is not None:
				semantics = np.asarray(semantics, dtype=np.int64)
				semantics = np.expand_dims(semantics, axis=2)
			if self.args.use_instance_segmentation is not None:
				instance_labels = np.asarray(instance_labels, dtype=np.int64)
				instance_labels = np.expand_dims(instance_labels, axis=2)
				instance_areas = np.asarray(instance_areas, dtype=np.int64)
				instance_areas = np.expand_dims(instance_areas, axis=2)

			if self.args.dataset == 'nyu':
				depth_gt = depth_gt / 1000.0
			else:
				depth_gt = depth_gt / 256.0

			if self.args.use_semantics is not None and self.args.use_instance_segmentation is not None:
				image, depth_gt, semantics, instance_labels, instance_areas = \
						self.random_crop_semantics_and_instance_segmentation(\
						image, depth_gt, semantics, instance_labels, instance_areas, \
						self.args.input_height, self.args.input_width)
				image, depth_gt, semantics, instance_labels, instance_areas = \
						self.train_preprocess_semantics_and_instance_segmentation(image, depth_gt, semantics, instance_labels, instance_areas)
				sample = {'image': image, 'depth': depth_gt, 'semantics': semantics, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}

			elif self.args.use_instance_segmentation is not None:
				image, depth_gt, instance_labels, instance_areas = self.random_crop_instance_segmentation( \
						image, depth_gt, instance_labels, instance_areas, \
						self.args.input_height, self.args.input_width)
				image, depth_gt, instance_labels, instance_areas = self.train_preprocess_instance_segmentation(image, depth_gt, instance_labels, instance_areas)
				sample = {'image': image, 'depth': depth_gt, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}

			elif self.args.use_semantics is not None:
				image, depth_gt, semantics = self.random_crop_semantics(image, depth_gt, semantics, self.args.input_height, self.args.input_width)
				image, depth_gt, semantics = self.train_preprocess_semantics(image, depth_gt, semantics)
				sample = {'image': image, 'depth': depth_gt, 'semantics': semantics, 'focal': focal}

			else:
				image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
				image, depth_gt = self.train_preprocess(image, depth_gt)
				sample = {'image': image, 'depth': depth_gt, 'focal': focal}

		else:
			if self.mode == 'online_eval':
				data_path = self.args.data_path_eval
			else:
				data_path = self.args.data_path

			image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
			image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

			if self.args.use_semantics is not None:
				if "ade20k-places" not in self.args.use_semantics:
					# NOT using ade20k-places classes (the instance segmentation ones)
					# => Using semantics from HRNetV2 (using the 150-class ADE20K subset)
					semantics_raw_path = image_path.replace('rgb', 'semantic_seg').replace('.jpg', '.npy')
					semantics_raw = np.load(semantics_raw_path)
				else:
					# Using ade20k-places classes (from instance segmentation model)
					# => Loading the results from the Swin-B Cascade Mask-RCNN model (which does instance segmentation)
					semantics_raw_path = image_path.replace('rgb', 'instance_labels_' + "ade20k_swin").replace('.jpg', '.npz')
					semantics_raw = np.load(semantics_raw_path, allow_pickle=True)
					semantics_raw = semantics_raw['arr_0']
					if len(semantics_raw.shape) != 2:
						semantics_raw = np.ones((image.size[1], image.size[0]), dtype=np.int32) * -1

				semantics = semantics_raw.astype(np.int64)
				semantics = np.expand_dims(semantics, axis=2)

			if self.args.use_instance_segmentation is not None:
				if "ade20k_swin" in self.args.use_instance_segmentation:
					# Using results from Swin-B Cascade Mask-RCNN model (instance segmentation)
					instance_labels_raw_path = image_path.replace('rgb', 'instance_labels_' + 'ade20k_swin').replace('.jpg', '.npz')
					if "bbox" in self.args.use_instance_segmentation:
						instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + 'ade20k_swin_bbox').replace('.jpg', '.npz')
					else:
						instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + 'ade20k_swin').replace('.jpg', '.npz')
				else:
					# Using some other instance segmentation model (likely coco)
					instance_labels_raw_path = image_path.replace('rgb', 'instance_labels_' + self.args.use_instance_segmentation).replace('.jpg', '.npy')
					instance_areas_raw_path = image_path.replace('rgb', 'instance_areas_' + self.args.use_instance_segmentation).replace('.jpg', '.npy')

				instance_labels_raw = np.load(instance_labels_raw_path, allow_pickle=True)
				instance_areas_raw = np.load(instance_areas_raw_path, allow_pickle=True)
				
				# Handle npz files differently
				if "ade20k_swin" in self.args.use_instance_segmentation:
					instance_labels_raw = instance_labels_raw['arr_0']
					instance_areas_raw = instance_areas_raw['arr_0']

					# If either of these are None, then Swin output no predictions for that image.
					# In this case, replace with default empty arrays (0 for area, -1 for labels)
					if len(instance_labels_raw.shape) != 2:
						instance_labels_raw = np.ones((image.size[1], image.size[0]), dtype=np.int32) * -1
					if len(instance_areas_raw.shape) != 2:
						instance_areas_raw = np.zeros((image.size[1], image.size[0]), dtype=np.int32)

				instance_labels = instance_labels_raw.astype(dtype=np.int64)
				instance_areas = instance_areas_raw
				instance_labels = np.expand_dims(instance_labels, axis=2)
				instance_areas = np.expand_dims(instance_areas, axis=2)


			if self.mode == 'online_eval':
				gt_path = self.args.gt_path_eval
				depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
				has_valid_depth = False
				try:
					depth_gt = Image.open(depth_path)
					has_valid_depth = True
				except IOError:
					depth_gt = False
					# print('Missing gt for {}'.format(image_path))

				if has_valid_depth:
					depth_gt = np.asarray(depth_gt, dtype=np.float32)
					depth_gt = np.expand_dims(depth_gt, axis=2)
					if self.args.dataset == 'nyu':
						depth_gt = depth_gt / 1000.0
					else:
						depth_gt = depth_gt / 256.0

			if self.args.do_kb_crop is True:
				height = image.shape[0]
				width = image.shape[1]
				top_margin = int(height - 352)
				left_margin = int((width - 1216) / 2)
				image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
				if self.args.use_semantics is not None:
					semantics = semantics[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
				if self.args.use_instance_segmentation is not None:
					instance_labels = instance_labels[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
					instance_areas = instance_areas[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
				if self.mode == 'online_eval' and has_valid_depth:
					depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

			if self.mode == 'online_eval':
				if self.args.use_semantics is not None and self.args.use_instance_segmentation is not None:
					sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'semantics': semantics, 
							'instance_labels': instance_labels, 'instance_areas': instance_areas,
							'has_valid_depth': has_valid_depth, 'image_path': sample_path.split()[0], 
							'depth_path': sample_path.split()[1]}
				elif self.args.use_instance_segmentation is not None:
					sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'instance_labels': instance_labels,
							'instance_areas': instance_areas,
							'has_valid_depth': has_valid_depth, 'image_path': sample_path.split()[0], 
							'depth_path': sample_path.split()[1]}
				elif self.args.use_semantics is not None:
					sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'semantics': semantics, 
							'has_valid_depth': has_valid_depth, 'image_path': sample_path.split()[0], 
							'depth_path': sample_path.split()[1]}
				else:
					sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
							  'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
			else:
				if self.args.use_semantics is not None and self.args.use_instance_segmentation is not None:
					sample = {'image': image, 'semantics': semantics, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}
				elif self.args.use_instance_segmentation is not None:
					sample = {'image': image, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}
				elif self.args.use_semantics is not None:
					sample = {'image': image, 'semantics': semantics, 'focal': focal}
				else:
					sample = {'image': image, 'focal': focal}

		# Slightly wasteful, but if requested then at this point, replace image with uniformly distributed noise.
		# We also ensure that noise is normalised as if it were an image, so image and noise values are in same range.
		if self.args.image == "noise":	# In this case, overwrite with uniformly distributed nonsense.
			sample['image'] = torch.rand(sample['image'].shape).numpy()	# Conversion to numpy to avoid changing transforms.

		if self.transform:
			sample = self.transform(sample)

		return sample

	def rotate_image(self, image, angle, flag=Image.BILINEAR):
		result = image.rotate(angle, resample=flag)
		return result

	# The next four functions should be merged into one.
	def random_crop(self, img, depth, height, width):
		assert img.shape[0] >= height
		assert img.shape[1] >= width
		assert img.shape[0] == depth.shape[0]
		assert img.shape[1] == depth.shape[1]
		x = random.randint(0, img.shape[1] - width)
		y = random.randint(0, img.shape[0] - height)
		img = img[y:y + height, x:x + width, :]
		depth = depth[y:y + height, x:x + width, :]
		return img, depth

	def random_crop_semantics(self, img, depth, semantics, height, width):
		assert img.shape[0] >= height
		assert img.shape[1] >= width
		assert img.shape[0] == depth.shape[0]
		assert img.shape[1] == depth.shape[1]
		assert img.shape[0] == semantics.shape[0]
		assert img.shape[1] == semantics.shape[1]
		x = random.randint(0, img.shape[1] - width)
		y = random.randint(0, img.shape[0] - height)
		img = img[y:y + height, x:x + width, :]
		depth = depth[y:y + height, x:x + width, :]
		semantics = semantics[y:y + height, x:x + width, :]
		return img, depth, semantics

	def random_crop_instance_segmentation(self, img, depth, instance_labels, instance_areas, height, width):
		assert img.shape[0] >= height
		assert img.shape[1] >= width
		assert img.shape[0] == depth.shape[0]
		assert img.shape[1] == depth.shape[1]
		assert img.shape[0] == instance_labels.shape[0]
		assert img.shape[1] == instance_labels.shape[1]
		assert img.shape[0] == instance_areas.shape[0]
		assert img.shape[1] == instance_areas.shape[1]
		x = random.randint(0, img.shape[1] - width)
		y = random.randint(0, img.shape[0] - height)
		img = img[y:y + height, x:x + width, :]
		depth = depth[y:y + height, x:x + width, :]
		instance_labels = instance_labels[y:y + height, x:x + width, :]
		instance_areas = instance_areas[y:y + height, x:x + width, :]
		return img, depth, instance_labels, instance_areas

	def random_crop_semantics_and_instance_segmentation(self, img, depth, semantics, instance_labels, instance_areas, height, width):
		assert img.shape[0] >= height
		assert img.shape[1] >= width
		assert img.shape[0] == depth.shape[0]
		assert img.shape[1] == depth.shape[1]
		assert img.shape[0] == semantics.shape[0]
		assert img.shape[1] == semantics.shape[1]
		assert img.shape[0] == instance_labels.shape[0]
		assert img.shape[1] == instance_labels.shape[1]
		assert img.shape[0] == instance_areas.shape[0]
		assert img.shape[1] == instance_areas.shape[1]
		x = random.randint(0, img.shape[1] - width)
		y = random.randint(0, img.shape[0] - height)
		img = img[y:y + height, x:x + width, :]
		depth = depth[y:y + height, x:x + width, :]
		semantics = semantics[y:y + height, x:x + width, :]
		instance_labels = instance_labels[y:y + height, x:x + width, :]
		instance_areas = instance_areas[y:y + height, x:x + width, :]
		return img, depth, semantics, instance_labels, instance_areas
	
	# 4 more functions that need to be merged into one.
	def train_preprocess(self, image, depth_gt):
		# Random flipping
		do_flip = random.random()
		if do_flip > 0.5:
			image = (image[:, ::-1, :]).copy()
			depth_gt = (depth_gt[:, ::-1, :]).copy()

		# Random gamma, brightness, color augmentation
		do_augment = random.random()
		if do_augment > 0.5:
			image = self.augment_image(image)

		return image, depth_gt

	def train_preprocess_semantics(self, image, depth_gt, semantics):
		# Random flipping
		do_flip = random.random()
		if do_flip > 0.5:
			image = (image[:, ::-1, :]).copy()
			depth_gt = (depth_gt[:, ::-1, :]).copy()
			semantics = (semantics[:, ::-1, :]).copy()

		# Random gamma, brightness, color augmentation
		do_augment = random.random()
		if do_augment > 0.5:
			image = self.augment_image(image)

		return image, depth_gt, semantics

	def train_preprocess_instance_segmentation(self, image, depth_gt, instance_labels, instance_areas):
		# Random flipping
		do_flip = random.random()
		if do_flip > 0.5:
			image = (image[:, ::-1, :]).copy()
			depth_gt = (depth_gt[:, ::-1, :]).copy()
			instance_labels = (instance_labels[:, ::-1, :]).copy()
			instance_areas = (instance_areas[:, ::-1, :]).copy()

		# Random gamma, brightness, color augmentation
		do_augment = random.random()
		if do_augment > 0.5:
			image = self.augment_image(image)

		return image, depth_gt, instance_labels, instance_areas

	def train_preprocess_semantics_and_instance_segmentation(self, image, depth_gt, semantics, instance_labels, instance_areas):
		# Random flipping
		do_flip = random.random()
		if do_flip > 0.5:
			image = (image[:, ::-1, :]).copy()
			depth_gt = (depth_gt[:, ::-1, :]).copy()
			semantics = (semantics[:, ::-1, :]).copy()
			instance_labels = (instance_labels[:, ::-1, :]).copy()
			instance_areas = (instance_areas[:, ::-1, :]).copy()

		# Random gamma, brightness, color augmentation
		do_augment = random.random()
		if do_augment > 0.5:
			image = self.augment_image(image)

		return image, depth_gt, semantics, instance_labels, instance_areas


	def augment_image(self, image):
		# gamma augmentation
		gamma = random.uniform(0.9, 1.1)
		image_aug = image ** gamma

		# brightness augmentation
		if self.args.dataset == 'nyu':
			brightness = random.uniform(0.75, 1.25)
		else:
			brightness = random.uniform(0.9, 1.1)
		image_aug = image_aug * brightness

		# color augmentation
		colors = np.random.uniform(0.9, 1.1, size=3)
		white = np.ones((image.shape[0], image.shape[1]))
		color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
		image_aug *= color_image
		image_aug = np.clip(image_aug, 0, 1)

		return image_aug

	def __len__(self):
		return len(self.filenames)


class ToTensor(object):
	def __init__(self, mode):
		self.mode = mode
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def __call__(self, sample):
		image, focal = sample['image'], sample['focal']
		image = self.to_tensor(image)
		image = self.normalize(image)
		if 'semantics' in sample:
			semantics = sample['semantics']
			semantics = self.to_tensor(semantics)
		if 'instance_labels' in sample:
			instance_labels = sample['instance_labels']
			instance_labels = self.to_tensor(instance_labels)
		if 'instance_areas' in sample:
			instance_areas = sample['instance_areas']
			instance_areas = self.to_tensor(instance_areas)

		if self.mode == 'test':
			if 'semantics' in sample and 'instance_labels' in sample:
				return {'image': image, 'semantics': semantics, 'instance_labels': instance_labels, 
						'instance_areas': instance_areas, 'focal': focal}
			elif 'instance_labels' in sample:
				return {'image': image, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}
			elif 'semantics' in sample:
				return {'image': image, 'semantics': semantics, 'focal': focal}
			else:
				return {'image': image, 'focal': focal}

		depth = sample['depth']
		if self.mode == 'train':
			depth = self.to_tensor(depth)

			if 'semantics' in sample and 'instance_labels' in sample:
				return {'image': image, 'depth': depth, 'semantics': semantics, 'instance_labels': instance_labels, 
						'instance_areas': instance_areas, 'focal': focal}
			elif 'instance_labels' in sample:
				return {'image': image, 'depth': depth, 'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal}
			elif 'semantics' in sample:
				return {'image': image, 'depth': depth, 'semantics': semantics, 'focal': focal}
			else:
				return {'image': image, 'depth': depth, 'focal': focal}

		else:
			has_valid_depth = sample['has_valid_depth']
			if 'semantics' in sample and 'instance_labels' in sample:
				return {'image': image, 'depth': depth, 'semantics': semantics, 
						'instance_labels': instance_labels, 'instance_areas': instance_areas, 'focal': focal, 
						'has_valid_depth': has_valid_depth, 'image_path': sample['image_path'],
						'depth_path': sample['depth_path']}
			elif 'instance_labels' in sample:
				return {'image': image, 'depth': depth, 'instance_labels': instance_labels, 
						'instance_areas': instance_areas, 'focal': focal, 
						'has_valid_depth': has_valid_depth, 'image_path': sample['image_path'],
						'depth_path': sample['depth_path']}
			elif 'semantics' in sample:
				return {'image': image, 'depth': depth, 'semantics': semantics, 'focal': focal, 
						'has_valid_depth': has_valid_depth, 'image_path': sample['image_path'],
						'depth_path': sample['depth_path']}
			else:
				return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
						'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

	def to_tensor(self, pic):
		if not (_is_pil_image(pic) or _is_numpy_image(pic)):
			raise TypeError(
				'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

		if isinstance(pic, np.ndarray):
			img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())
			return img

		# handle PIL Image
		if pic.mode == 'I':
			img = torch.from_numpy(np.array(pic, np.int32, copy=False))
		elif pic.mode == 'I;16':
			img = torch.from_numpy(np.array(pic, np.int16, copy=False))
		else:
			img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
		# PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
		if pic.mode == 'YCbCr':
			nchannel = 3
		elif pic.mode == 'I;16':
			nchannel = 1
		else:
			nchannel = len(pic.mode)
		img = img.view(pic.size[1], pic.size[0], nchannel)

		img = img.transpose(0, 1).transpose(0, 2).contiguous()
		if isinstance(img, torch.ByteTensor):
			return img.float()
		else:
			return img
