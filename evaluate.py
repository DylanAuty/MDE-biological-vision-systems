import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
import utils
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict

from ArgParseWrappers.EvalArgParser import EvalArgParser
from ExternalInfoLoaders.SemanticsLoader import SemanticsLoader
from ExternalInfoLoaders.InstanceSegmentationLoader import InstanceSegmentationLoader

# Loading glove embeddings
glove_embeddings = np.load('data/ade20k_150_classes_glove_840b_300d_embeddings.npy')
glove_embeddings = torch.from_numpy(glove_embeddings)

"""
def compute_errors(gt, pred):
	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean(np.abs(gt - pred) / gt)
	sq_rel = np.mean(((gt - pred) ** 2) / gt)

	rmse = (gt - pred) ** 2
	rmse = np.sqrt(rmse.mean())

	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	err = np.log(pred) - np.log(gt)
	silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

	log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
	return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
				silog=silog, sq_rel=sq_rel)
"""


def predict_tta(model, gt, image, args, semantics=None, instance_labels=None, instance_areas=None):
	if semantics is not None and instance_labels is not None:
		bin_edges, pred = model(x=image, semantics=semantics, instance_labels=instance_labels, instance_areas=instance_areas)
	elif instance_labels is not None:
		bin_edges, pred = model(x=image, instance_labels=instance_labels, instance_areas=instance_areas)
	elif semantics is not None:
		bin_edges, pred = model(x=image, semantics=semantics)
	else:
		bin_edges, pred = model(image)

	mask = gt > args.min_depth
	pred = nn.functional.interpolate(pred, gt.shape[-3:-1], mode='bilinear', align_corners=True)
	pred = pred.squeeze().cpu().numpy()

	pred[pred < args.min_depth_eval] = args.min_depth_eval
	pred[pred > args.max_depth_eval] = args.max_depth_eval
	pred[np.isinf(pred)] = args.max_depth_eval
	pred[np.isnan(pred)] = args.min_depth_eval

	gt_depth = gt.squeeze().cpu().numpy()
	valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
	return pred


def eval(model, test_loader, args, gpus=None, ):
	if gpus is None:
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	else:
		device = gpus[0]

	if args.save_dir is not None:
		os.makedirs(args.save_dir)

	metrics = RunningAverageDict()
	# crop_size = (471 - 45, 601 - 41)
	# bins = utils.get_bins(100)
	total_invalid = 0
	with torch.no_grad():
		model.eval()

		sequential = test_loader

		semantics_loader = SemanticsLoader(args)			# Loads word embeddings and sizes for semantic labels
		instance_loader = InstanceSegmentationLoader(args)	# Responsible for loading class and humans-size embeddings based on the
															# instance segmentation model's results
		for batch in tqdm(sequential):
			image = batch['image'].to(device)
			gt = batch['depth'].to(device)

			# Get embedded and raw semantics. If semantics aren't being used, these will both be None.
			semantics_raw, semantics = semantics_loader.get_semantics(batch)

			# Get embedded and raw instance labels, and per-instance areas. Will be None if instance segmentation is not being used.
			instance_labels_raw, instance_labels, instance_areas = instance_loader.get_instance_segmentation(batch)

			if args.use_semantics is not None and args.use_instance_segmentation is not None:
				final = predict_tta(model, gt, image, args, semantics=semantics, instance_labels=instance_labels, instance_areas=instance_areas)
			elif args.use_semantics is not None:
				final = predict_tta(model, gt, image, args, semantics=semantics)
			elif args.use_instance_segmentation is not None:
				final = predict_tta(model, gt, image, args, instance_labels=instance_labels, instance_areas=instance_areas)
			else:
				final = predict_tta(model, gt, image, args)

			if args.save_dir is not None:
				if args.dataset == 'nyu':
					impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
					factor = 1000
				else:
					dpath = batch['image_path'][0].split('/')
					impath = dpath[1] + "_" + dpath[-1]
					impath = impath.split('.')[0]
					factor = 256

				pred_path = os.path.join(args.save_dir, f"{impath}.png")
				pred = (final * factor).astype('uint16')
				Image.fromarray(pred).save(pred_path)

			if 'has_valid_depth' in batch:
				if not batch['has_valid_depth']:
					total_invalid += 1
					continue

			gt = gt.squeeze().cpu().numpy()
			valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

			if args.garg_crop or args.eigen_crop:
				gt_height, gt_width = gt.shape
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

			metrics.update(utils.compute_errors(gt[valid_mask], final[valid_mask]))

	print(f"Total invalid: {total_invalid}")
	metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
	print(f"Metrics: {metrics}")


if __name__ == '__main__':

	# Arguments are all handled in EvalArgParser (most of them in the parent class CommonArgParser)
	parser = EvalArgParser(description='Model evaluator', fromfile_prefix_chars='@', conflict_handler='resolve')

	if sys.argv.__len__() == 2:
		arg_filename_with_prefix = '@' + sys.argv[1]
		args = parser.parse_args([arg_filename_with_prefix])
	else:
		args = parser.parse_args()

	args.gpu = int(args.gpu) if args.gpu is not None else 0
	args.distributed = False
	device = torch.device('cuda:{}'.format(args.gpu))
	test = DepthDataLoader(args, 'online_eval').data
	model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
								   norm='linear', encoder_name=args.encoder_name, semantics_mode=args.use_semantics, instance_segmentation_mode=args.use_instance_segmentation, insertion_point=args.insertion_point, image=args.image).to(device)
	model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
	model = model.eval()

	eval(model, test, args, gpus=[device])
