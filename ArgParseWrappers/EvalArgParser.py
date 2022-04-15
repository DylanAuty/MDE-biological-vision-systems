# EvalArgParser.py
# Implements EvalArgParser, which handles evaluation-specific command line arguments.

import argparse
from .CommonArgParser import CommonArgParser

class EvalArgParser(CommonArgParser):
	def __init__(self, description="Evaluation script argument parser class", fromfile_prefix_chars='@', conflict_handler='resolve'):
		super().__init__(description=description, fromfile_prefix_chars=fromfile_prefix_chars, conflict_handler=conflict_handler)

		# Model setup - n_bins is repeated in both train and eval arg parsers, but with different default values.
		self.add_argument('--n-bins', '--n_bins', default=256, type=int, help='number of bins/buckets to divide depth range into')

		# Path setup
		self.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
		self.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
							help="checkpoint file to use for prediction")

		# This is in both TrainArgParser and EvalArgParser, except one has default=True and the other doesn't.
		self.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
