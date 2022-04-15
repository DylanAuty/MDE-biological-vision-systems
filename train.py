import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
from scipy.io import loadmat
import colorcet as cc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize
from GraphBuilder_NYUD2 import GraphBuilder_NYUD2

import matplotlib

from ArgParseWrappers.TrainArgParser import TrainArgParser
from ExternalInfoLoaders.SemanticsLoader import SemanticsLoader
from ExternalInfoLoaders.InstanceSegmentationLoader import InstanceSegmentationLoader


logging = True


def is_rank_zero(args):
	return args.rank == 0

# Loading semantic colours
colors = loadmat('data/color150.mat')['colors']

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: coco_class_names.index('teddy bear')
# These come from github.com/matterport/Mask_RCNN. Their Dataset class overrides
# the original class numbers, giving the array below. To retrieve this otherwise,
# the COCO dataset must be downloaded and loaded.
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

# Absolute sizes of ade20k classes, aligned with class names as in ade20k_classes array above,
# in order Height, Width, Length. Very approximate measurements. In metres.
# Width generally refers to the "front" of an object, e.g. the front of a fridge or the short edge of a bed.
# This copy commented out so that it doesn't interfere with the loading of the same thing from a .npy file.
"""
ade20k_classes_abs_sizes = [[0.97, 1.37, 2.0], [1.0, 0.915, 0.14], [0.85, 0.6, 0.6], [1.74, 0.465, 0.25], [1.981, 0.762, 0.035],
		[0.76, 0.85, 1.8], [1.6, 1.4, 0.035], [1.060, 0.6, 0.6], [1.46, 1.7, 4.5], [0.594, 0.420, 0.015], 
		[0.88, 1.85, 0.85], [0.025, 0.9, 0.3], [0.594, 0.420, 0.01], [0.88, 0.85, 0.85], [0.81, 0.42, 0.45],
		[1.5, 1.8, 0.09], [0.62, 1.4, 0.65], [1.9, 1.17, 0.5], [1.0, 0.4, 0.4], [0.425, 0.6, 1.52],
		[0.9, 0.08, 4.0], [0.43, 0.43, 0.1], [0.35, 0.4, 0.3], [2.5, 0.3, 0.3], [0.91, 0.601, 0.01],
		[1.0, 1.08, 0.5], [2.0, 1.7, 0.85], [0.15, 0.55, 0.35], [1.13, 1.28, 0.16], [1.68, 0.762, 0.81],
		[2.4, 1.0, 3.0], [0.67, 0.45, 0.3], [0.813, 0.15, 2.62], [0.11, 0.66, 0.508], [1.981, 0.762, 0.035],
		[2.12, 1.1, 0.37], [0.45, 0.55, 0.90], [0.762, 0.56, 0.62], [0.2, 0.07, 0.07], [0.197, 0.129, 0.04],
		[0.81, 1.524, 0.76], [0.85, 1.2, 0.01], [0.85, 0.6, 0.6], [1.06, 0.91, 0.91], [0.85, 1.2, 0.6],
		[0.45, 0.2, 0.55], [0.87, 0.62, 0.62], [0.61, 1.3, 3.7], [1.841, 0.5, 0.6], [2.99, 2.55, 11.95],
		[1.27, 63.5, 0.01], [0.4, 0.38, 0.38], [1.92, 2.03, 5.89], [1.0, 0.9, 0.9], [1.5, 2.0, 1.5],
		[0.7, 1.0, 0.2], [2.164, 0.85, 0.85], [0.08, 0.44, 0.29], [18.97, 64.4, 68.6], [0.52, 0.465, 0.25],
		[2.0, 0.17, 0.17], [1.3, 0.08, 2.5], [0.43, 0.66, 0.66], [0.3, 0.1, 0.1], [2.54, 2.7, 5.57],
		[14.5, 40.0, 300.0], [0.89, 1.0, 1.0], [0.85, 0.6, 0.6], [0.2, 0.2, 0.2], [0.81, 0.35, 0.35],
		[0.88, 0.61, 0.61], [0.225, 0.334, 0.487], [0.35, 0.27, 0.2], [0.5, 0.3, 0.9], [0.85, 0.6, 0.650],
		[0.225, 0.225, 0.225], [0.06, 0.15, 0.15], [0.165, 0.86, 0.26], [0.5, 2.0, 0.05], [0.311, 0.525, 0.402],
		[0.13, 0.4, 0.4], [0.40, 0.17, 0.60], [1.0, 0.5, 1.69], [0.85, 0.6, 0.625], [0.27, 0.44, 0.03],
		[0.6, 0.6, 0.6], [0.292, 0.762, 0.508], [0.43, 0.21, 0.21], [0.25, 0.18, 0.18], [0.75, 0.205, 0.305],
		[0.0178, 0.414, 0.306], [0.647, 0.432, 0.33], [0.1, 0.41, 0.41], [0.02, 0.2667, 0.2667], [0.35, 0.55, 0.02],
		[0.9, 1.2, 0.01], [0.58, 1.01, 0.13], [0.12, 0.08, 0.08], [0.28, 0.28, 0.04], [0.762, 1.2192, 0.01],
		[-1, -1, -1]]
"""


def visualize_semantics(semantics):
	"""Visualises a semantics segmentation map using the classes from ADE20K.
	Function based on one from https://github.com/CSAILVision/semantic-segmentation-pytorch
	
	Expects a batched input (Bx1xHxW).
	"""
	semantics = np.int32(semantics)

	"""
	# print predictions in descending order
	pixs = semantics.size
	uniques, counts = np.unique(semantics, return_counts=True)
	for idx in np.argsort(counts)[::-1]:
		name = names[uniques[idx] + 1]
		ratio = counts[idx] / pixs * 100
		if ratio > 0.1:
			print("  {}: {:.2f}%".format(name, ratio))
	"""

	# colorize prediction
	semantics_color = np.zeros([semantics.shape[0], 3, semantics.shape[2], semantics.shape[3]], dtype=semantics.dtype)
	for i in range(0, semantics.shape[0]):
		semantics_color[i] = colorEncode(semantics[i], colors).astype(np.uint8)

	return semantics_color


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
	"""Used by the colorEncode function to work out the number of unique labels in a semantics segmentation map.
	Taken from https://github.com/CSAILVision/semantic-segmentation-pytorch
	"""
	ar = np.asanyarray(ar).flatten()

	optional_indices = return_index or return_inverse
	optional_returns = optional_indices or return_counts

	if ar.size == 0:
		if not optional_returns:
			ret = ar
		else:
			ret = (ar,)
			if return_index:
				ret += (np.empty(0, np.bool),)
			if return_inverse:
				ret += (np.empty(0, np.bool),)
			if return_counts:
				ret += (np.empty(0, np.intp),)
		return ret
	if optional_indices:
		perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
		aux = ar[perm]
	else:
		ar.sort()
		aux = ar
	flag = np.concatenate(([True], aux[1:] != aux[:-1]))

	if not optional_returns:
		ret = aux[flag]
	else:
		ret = (aux[flag],)
		if return_index:
			ret += (perm[flag],)
		if return_inverse:
			iflag = np.cumsum(flag) - 1
			inv_idx = np.empty(ar.shape, dtype=np.intp)
			inv_idx[perm] = iflag
			ret += (inv_idx,)
		if return_counts:
			idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
			ret += (np.diff(idx),)
	return ret


def colorEncode(labelmap, colors, mode='RGB'):
	"""Colourises a semantic map according to the colours passed to it.
	From https://github.com/CSAILVision/semantic-segmentation-pytorch	

	Expects a non-batched input (1xHxW)
	"""
	labelmap = np.transpose(labelmap.astype('int'), axes=[1, 2, 0])	# Input is 1xHxW but this fn needs HxWx1.
	labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
							dtype=np.uint8)
	for label in unique(labelmap):
		if label < 0:
			continue
		labelmap_rgb += (labelmap == label) * \
			np.tile(colors[label],
					(labelmap.shape[0], labelmap.shape[1], 1))

	if mode == 'BGR':
		return np.transpose(labelmap_rgb[:, :, ::-1], [2, 0, 1])
	else:
		return np.transpose(labelmap_rgb, [2, 0, 1])


def colorize_batched(value, vmin=10, vmax=1000, cmap='magma_r'):
	"""
	:param value: torch.Tensor(Bx1xHxW)
	
	returns img (torch.Tensor, Bx3xHxW)
	"""
	cmapper = cc.m_bkr
	val_clone = value.clone().detach().cpu()
	img = value.clone().expand(-1, 3, -1, -1).clone()

	for i in range(0, val_clone.shape[0]):
		# normalize
		vmin = val_clone[i].min() if vmin is None else vmin
		vmax = val_clone[i].max() if vmax is None else vmax
		if vmin != vmax:
			val_clone[i] = (val_clone[i] - vmin) / (vmax - vmin)  # vmin..vmax
		else:
			# Avoid 0-division
			val_clone[i] = val_clone[i] * 0.
		# squeeze last dim if it exists
		# value = value.squeeze(axis=0)

		tmp = cmapper(val_clone[i], bytes=True)  # (nxmx4)
		tmp = torch.from_numpy(tmp)

		img[i] = tmp[0, :, :, :3].permute(2, 0, 1)
		

	return img


def log_images_batched(img, depth, pred, args, writer, train, step):
	"""
	:param img: Torch.Tensor (Bx3xHxW)
	:param depth: Torch.Tensor (Bx1xHxW)
	:param pred: Torch.Tensor (Bx1xHxW)
	:param writer: SummaryWriter.
	:param train: Boolean. True if train, False if test. Affects the label for the images.
	"""
	depth = colorize_batched(depth, vmin=args.min_depth, vmax=args.max_depth)
	pred = colorize_batched(pred, vmin=args.min_depth, vmax=args.max_depth)
	print(depth.shape)
	print(pred.shape)

	label_string = 'Train' if train else 'Test'
	writer.add_images(f'{label_string}/Input', img, step)
	writer.add_images(f'{label_string}/GT', depth, step)
	writer.add_images(f'{label_string}/Prediction', pred, step)

	return


def main_worker(gpu, ngpus_per_node, args):
	print(f"Main worker: gpu = {gpu}, ngpus_per_node = {ngpus_per_node}")
	args.gpu = gpu

	###################################### Load model ##############################################

	model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
										  norm=args.norm, encoder_name=args.encoder_name, semantics_mode=args.use_semantics, instance_segmentation_mode=args.use_instance_segmentation, insertion_point=args.insertion_point, image=args.image)

	################################################################################################

	if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)

	args.multigpu = False
	if args.distributed:
		# Use DDP
		args.multigpu = True
		args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
		if args.use_new_batching:
			print("Using new batching (batch_size examples per GPU)")
		else:
			# Original code split one batch over all GPUs, which caused strange things to happen with many GPUs.
			print("Using original batching strategy (split batch across available GPUs)")
			args.batch_size = int(args.batch_size / ngpus_per_node)

		args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
		print(f"WORKER [{gpu}]: GPU={args.gpu}, rank={args.rank}, batch_size={args.batch_size}, workers={args.workers}")
		torch.cuda.set_device(args.gpu)
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = model.cuda(args.gpu)
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
														  find_unused_parameters=True)

	elif args.gpu is None:
		# Use DP
		args.multigpu = True
		model = model.cuda()
		model = torch.nn.DataParallel(model)

	args.epoch = 0
	args.last_epoch = -1
	train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
		  experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
		  optimizer_state_dict=None):
	if device is None:
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	###################################### Logging setup #########################################
	print(f"Training {experiment_name}")

	run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
	name = f"{experiment_name}_{run_id}"
	should_write = ((not args.distributed) or args.rank == 0)
	should_log = should_write and logging
	if should_log:
		tags = args.tags.split(',') if args.tags != '' else None
		writer = SummaryWriter(args.exp_dir, flush_secs=1)
		graphBuilder = GraphBuilder_NYUD2(num_samples=2, semantics=(args.use_semantics is not None), instance_labels=(args.use_instance_segmentation is not None))
	################################################################################################

	# Creating the DepthDataLoader (which is not itself a torch dataloader) so I can access the sampler later.
	train_loader_cls = DepthDataLoader(args, 'train')
	train_loader = train_loader_cls.data
	test_loader_cls = DepthDataLoader(args, 'online_eval')
	test_loader = test_loader_cls.data

	###################################### losses ##############################################
	criterion_ueff = SILogLoss()
	criterion_bins = BinsChamferLoss() if args.chamfer else None
	################################################################################################

	model.train()

	###################################### Optimizer ################################################
	if args.same_lr:
		print("Using same LR")
		params = model.parameters()
	else:
		print("Using diff LR")
		m = model.module if args.multigpu else model
		params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
				  {"params": m.get_10x_lr_params(), "lr": lr}]

	optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
	if optimizer_state_dict is not None:
		optimizer.load_state_dict(optimizer_state_dict)
	################################################################################################
	# some globals
	iters = len(train_loader)
	step = args.epoch * iters
	best_loss = np.inf

	###################################### Scheduler ###############################################
	scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
											  cycle_momentum=True,
											  base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
											  div_factor=args.div_factor,
											  final_div_factor=args.final_div_factor)
	if args.resume != '' and scheduler is not None:
		scheduler.step(args.epoch + 1)
	################################################################################################

	semantics_loader = SemanticsLoader(args)	# Responsible for loading external information that ends up in the semantics tensor
	instance_loader = InstanceSegmentationLoader(args)	# Responsible for loading class and humans-size embeddings based on the
														# instance segmentation model's results

	# max_iter = len(train_loader) * epochs
	for epoch in range(args.epoch, epochs):
		################################# Train loop ##########################################################
		if should_log: writer.add_scalar('Epoch', epoch, step)
		ade20k_classes_abs_sizes = None

		if args.distributed:
			# This is needed to make the shuffling work with Pytorch DistributedDataParallel.
			train_loader_cls.train_sampler.set_epoch(epoch)

		for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
							 total=len(train_loader)) if is_rank_zero(
				args) else enumerate(train_loader):

			optimizer.zero_grad()

			img = batch['image'].to(device)		# Regardless of image setting, we always load this for use in graphs.
			depth = batch['depth'].to(device)
			if 'has_valid_depth' in batch:
				if not batch['has_valid_depth']:
					continue
			
			# Get embedded and raw semantics. If semantics aren't being used, these will both be None.
			semantics_raw, semantics = semantics_loader.get_semantics(batch)

			# Get embedded and raw instance labels, and per-instance areas. Will be None if instance segmentation is not being used.
			instance_labels_raw, instance_labels, instance_areas = instance_loader.get_instance_segmentation(batch)

			if args.use_semantics is not None and args.use_instance_segmentation is not None:
				bin_edges, pred = model(x=img, semantics=semantics, instance_labels=instance_labels, instance_areas=instance_areas)
			elif args.use_instance_segmentation is not None:
				bin_edges, pred = model(x=img, instance_labels=instance_labels, instance_areas=instance_areas)
			elif args.use_semantics is not None:
				bin_edges, pred = model(x=img, semantics=semantics)
			else:
				bin_edges, pred = model(img)

			mask = depth > args.min_depth
			l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

			if args.w_chamfer > 0:
				if bin_edges is not None:
					l_chamfer = criterion_bins(bin_edges, depth)
				else:
					l_chamfer = torch.Tensor([0]).to(img.device)	# If there's no AdaBins module then l_chamfer makes no sense.
			else:
				l_chamfer = torch.Tensor([0]).to(img.device)

			loss = l_dense + args.w_chamfer * l_chamfer
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
			optimizer.step()

			# Logging: Every 5 steps log loss, every 500 log some qualitative samples, every args.validate_every run
			# validation and save checkpoints.
			if should_log and step % 5 == 0:
				writer.add_scalar('Train/Loss', loss.item(), step)
				writer.add_scalar(f'Train/{criterion_ueff.name}', l_dense.item(), step)
				writer.add_scalar(f'Train/{criterion_bins.name}', l_chamfer.item(), step)

			if should_log and step % 500 == 0:
				#log_images_batched(img, depth, pred, args, writer, True, step)
				if args.use_semantics is not None and args.use_instance_segmentation is not None:
					semantics_color = visualize_semantics(semantics_raw)
					instance_labels_color = visualize_semantics(instance_labels_raw)	# visualize_semantics() works for this too.
					graphBuilder.add_image(img, depth, pred, semantics=semantics_color, instance_labels=instance_labels_color)
				elif args.use_instance_segmentation is not None:
					instance_labels_color = visualize_semantics(instance_labels_raw)	# visualize_semantics() works for this too.
					graphBuilder.add_image(img, depth, pred, instance_labels=instance_labels_color)
				elif args.use_semantics is not None:
					semantics_color = visualize_semantics(semantics_raw)
					graphBuilder.add_image(img, depth, pred, semantics=semantics_color)
				else:
					graphBuilder.add_image(img, depth, pred)
				writer.add_figure('Train/QualSamples', graphBuilder.fig, step)
				graphBuilder.reset()

			step += 1
			scheduler.step()

			########################################################################################################

			if should_write and step % args.validate_every == 0:

				################################# Validation loop ##################################################
				model.eval()
				if args.use_semantics is not None:
					metrics, val_si, val_img, val_depth, val_pred, val_semantics = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)
				else:
					metrics, val_si, val_img, val_depth, val_pred = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

				# print("Validated: {}".format(metrics))
				if should_log:
					writer.add_scalar(f'Test/{criterion_ueff.name}', val_si.get_value(), step)
					#writer.add_scalar(f'Test/{criterion_bins.name}', val_bins.get_value(), step)

					for k, v in metrics.items():
						writer.add_scalar(f'Metrics/{k}', v, step)

					if args.use_semantics is not None and args.use_instance_segmentation is not None:
						semantics_color = visualize_semantics(semantics_raw)
						instance_labels_color = visualize_semantics(instance_labels_raw)	# visualize_semantics() works for this too.
						graphBuilder.add_image(img, depth, pred, semantics=semantics_color, instance_labels=instance_labels_color)
					elif args.use_instance_segmentation is not None:
						instance_labels_color = visualize_semantics(instance_labels_raw)	# visualize_semantics() works for this too.
						graphBuilder.add_image(img, depth, pred, instance_labels=instance_labels_color)
					elif args.use_semantics is not None:
						semantics_color = visualize_semantics(semantics_raw)
						graphBuilder.add_image(img, depth, pred, semantics=semantics_color)
					else:
						graphBuilder.add_image(img, depth, pred)

					writer.add_figure('Test/QualSamples', graphBuilder.fig, step)
					graphBuilder.reset()

					model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
											 root=args.exp_dir)

				if metrics['abs_rel'] < best_loss and should_write:
					model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
											 root=args.exp_dir)
					best_loss = metrics['abs_rel']
				model.train()
				#################################################################################################

	return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
	with torch.no_grad():
		val_si = RunningAverage()
		# val_bins = RunningAverage()
		metrics = utils.RunningAverageDict()

		semantics_loader = SemanticsLoader(args)	# Responsible for loading external information that ends up in the semantics tensor
		instance_loader = InstanceSegmentationLoader(args)	# Responsible for loading class and humans-size embeddings based on the
															# instance segmentation model's results

		for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(args) else test_loader:
			img = batch['image'].to(device)
			depth = batch['depth'].to(device)
			if 'has_valid_depth' in batch:
				if not batch['has_valid_depth']:
					continue

			depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

			# Get embedded and raw semantics. If semantics aren't being used, these will both be None.
			semantics_raw, semantics = semantics_loader.get_semantics(batch)

			# Get embedded and raw instance labels, and per-instance areas. Will be None if instance segmentation is not being used.
			instance_labels_raw, instance_labels, instance_areas= instance_loader.get_instance_segmentation(batch)

			if args.use_semantics is not None and args.use_instance_segmentation is not None:
				bin_edges, pred = model(x=img, semantics=semantics, instance_labels=instance_labels, instance_areas=instance_areas)
			elif args.use_instance_segmentation is not None:
				bin_edges, pred = model(x=img, instance_labels=instance_labels, instance_areas=instance_areas)
			elif args.use_semantics is not None:
				bin_edges, pred = model(x=img, semantics=semantics)
			else:
				bin_edges, pred = model(img)

			mask = depth > args.min_depth
			l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
			val_si.append(l_dense.item())

			pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

			pred = pred.squeeze().cpu().numpy()
			pred[pred < args.min_depth_eval] = args.min_depth_eval
			pred[pred > args.max_depth_eval] = args.max_depth_eval
			pred[np.isinf(pred)] = args.max_depth_eval
			pred[np.isnan(pred)] = args.min_depth_eval

			gt_depth = depth.squeeze().cpu().numpy()
			valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
			if args.garg_crop or args.eigen_crop:
				gt_height, gt_width = gt_depth.shape
				eval_mask = np.zeros(valid_mask.shape)

				if args.garg_crop:
					eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
					int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

				elif args.eigen_crop:
					if args.dataset == 'kitti':
						eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
						int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
					else:
						eval_mask[45:471, 41:601] = 1
			valid_mask = np.logical_and(valid_mask, eval_mask)
			metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

	if args.use_semantics is not None:
		return metrics.get_value(), val_si, img, depth, pred, semantics
	else:
		return metrics.get_value(), val_si, img, depth, pred


if __name__ == '__main__':

	# Arguments (handled in TrainArgParser and in its parent class, CommonArgParser).
	parser = TrainArgParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@', conflict_handler='resolve')

	if sys.argv.__len__() == 2:
		arg_filename_with_prefix = '@' + sys.argv[1]
		args = parser.parse_args([arg_filename_with_prefix])
	else:
		args = parser.parse_args()

	args.batch_size = args.bs
	args.num_threads = args.workers
	args.mode = 'train'
	args.chamfer = args.w_chamfer > 0

	# To allow use of $HOME, $TMPDIR, etc.
	args.root = os.path.expandvars(args.root)
	args.data_path = os.path.expandvars(args.data_path)
	args.gt_path = os.path.expandvars(args.gt_path)
	args.data_path_eval = os.path.expandvars(args.data_path_eval)
	args.gt_path_eval = os.path.expandvars(args.gt_path_eval)

	if args.root != "." and not os.path.isdir(args.root):
		os.makedirs(args.root)

	args.exp_dir = utils.setUpExpDir(args.root, args.name)

	try:
		node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
		nodes = node_str.split(',')

		args.world_size = len(nodes)
		args.rank = int(os.environ['SLURM_PROCID'])

	except KeyError as e:
		# We are NOT using SLURM
		args.world_size = 1
		args.rank = 0
		nodes = ["127.0.0.1"]

	if args.distributed:
		print("Running distributed")
		mp.set_start_method('forkserver')

		print(f"Rank: {args.rank}")
		port = np.random.randint(15000, 15025)
		args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
		print(args.dist_url)
		args.dist_backend = 'nccl'
		args.gpu = None

	ngpus_per_node = torch.cuda.device_count()
	args.num_workers = args.workers
	args.ngpus_per_node = ngpus_per_node

	print(f"Distributed training: {args.distributed}")
	if args.distributed:
		args.world_size = ngpus_per_node * args.world_size
		print(f"ngpus_per_node: {ngpus_per_node}")
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		if ngpus_per_node == 1:
			args.gpu = 0
		main_worker(args.gpu, ngpus_per_node, args)
