# TrainArgParser.py
# Contains implementation for TrainArgParser, which contains training-specific command line arguments.

import argparse
from .CommonArgParser import CommonArgParser

class TrainArgParser(CommonArgParser):
	def __init__(self, description="Training script argument parser class.", fromfile_prefix_chars='@', conflict_handler='resolve'):
		super().__init__(description=description, fromfile_prefix_chars=fromfile_prefix_chars, conflict_handler=conflict_handler)
		
		# System setup
		self.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
		self.add_argument("--distributed", action="store_true", help="Use DDP if set")
		
		# Disused/never implemented things that I won't remove in case some part of the code needs them
		self.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
		self.add_argument("--notes", default='', type=str, help="Wandb notes")
		self.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

		# Experiment setup - basics
		self.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
		self.add_argument('--bs', default=16, type=int, help='batch size')
		self.add_argument("--use_new_batching", default=False, action="store_true", help="Whether to use the original batching strategy where batch_size examples were split across all available GPUs. If this is set to False, will put the same number of examples on each GPU (or will try to).")
		self.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
		self.add_argument("--name", default="UnetAdaptiveBins")

		self.add_argument('--n-bins', '--n_bins', default=80, type=int, help='number of bins/buckets to divide depth range into')

		# Experiment setup - loss weights, learning rate etc.
		self.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
		self.add_argument("--same-lr", '--same_lr', default=False, action="store_true", help="Use same LR for all param groups")
		self.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
		self.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float, help="final div factor for lr")
		self.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
		self.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")

		self.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
							choices=['linear', 'softmax', 'sigmoid'])

		# Experiment setup - data augmentation/cropping
		self.add_argument('--do_random_rotate', default=True,
							help='if set, will perform random rotation for augmentation',
							action='store_true')
		self.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
		self.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
							action='store_true')

		# This is in both TrainArgParser and EvalArgParser, except one has default=True and the other doesn't.
		self.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14', action='store_true')


if __name__ == "__main__":
	# Quick test code
	parser = TrainArgParser(description="Training Arg Parser")
	args = parser.parse_args()
	breakpoint()
