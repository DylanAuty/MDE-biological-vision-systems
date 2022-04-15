# CommonArgParser.py
# Contains class definition for CommonArgParser, a wrapper around argparse.ArgumentParser.
# This class contains arguments common to both training and evaluation scripts.

import argparse

class CommonArgParser(argparse.ArgumentParser):
	def __init__(self, description="Common Argument Parser base class.", fromfile_prefix_chars='@', conflict_handler='resolve'):
		# Init an argparser, slap in the arguments common to both train and test.
		super().__init__(description=description, fromfile_prefix_chars=fromfile_prefix_chars, conflict_handler=conflict_handler)

		# System setup (GPU, root directory) 
		self.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
		self.add_argument("--root", default=".", type=str,
						help="Root folder to save data in")

		# Dataset setup (Which to use, paths to various splits)
		self.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
		self.add_argument("--data_path", default='nyu/sync/', type=str, help="path to dataset")
		self.add_argument("--gt_path", default='nyu/sync/', type=str, help="path to dataset")
		self.add_argument('--filenames_file',
							default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
							type=str, help='path to the filenames text file')
		self.add_argument('--data_path_eval',
							default="nyu/official_splits/test/",
							type=str, help='path to the data for online evaluation')
		self.add_argument('--gt_path_eval', default="nyu/official_splits/test/",
							type=str, help='path to the groundtruth data for online evaluation')
		self.add_argument('--filenames_file_eval',
							default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
							type=str, help='path to the filenames text file for online evaluation')

		self.add_argument('--input_height', type=int, help='input height', default=416)
		self.add_argument('--input_width', type=int, help='input width', default=544)
		self.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
		self.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

		self.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
		self.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
		self.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
		self.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')


		self.add_argument("--use_semantics", default=None, type=str, help="""
				What kind of semantic information to use. Possible values are:
					- 'raw', \
					- 'glove', \
					- 'glove-25d', \
					- 'glove-25d-inst-areas' to include areas of each class, \
					- 'glove-25d-ade20k-places' for semantics from the instance segmentation model \
					- 	(trained on ADE20K Places Challenge classes (100 labels)), and \
					- 'glove-25d-ade20k-places-human-sizes' for the same but including absolute human-provided sizes per class. \
						- Adding \"size_shuffled\" to a glove-25d thing will use a shuffled version of the glove embeddings (swapping embeddings for different classes). \
						- Adding \"shuffled\" to a human-sizes thing will use class-shuffled true object sizes. \
					""")
		self.add_argument("--encoder_name", default="efficientnet-b5", type=str, help="""
				Either \"efficientnet-b5\" or \"efficientnet-b1\", or \"efficientnet-b1-noAdaBins\" for a basic bottleneck without the AdaBins module.\
				""")
		self.add_argument("--use_instance_segmentation", default=None, type=str, help="\
				Whether/what instance segmentation to use. Default is \"None\". Can also be:\
				\"coco\" to work with predictions from Mask-RCNN trained on MS-COCO, with MS-COCO class names, \
				\"ade20k_swin\" to use predictions from a Swin transformer trained on the ADE20K dataset, or \
				\"ade20k_swin_human_sizes\" for human-labeled absolute sizes in addition to per-instance visual area. \
				Can also be \"ade20k_swin_bbox\" or \"ade20k_swin_bbox_human_sizes\" to use instance bounding box areas instead of instance mask areas.")
		self.add_argument("--insertion_point", default="before-attn", type=str, help="Where to insert semantic segmentation or instance segmentation/instance area information. Default is \"before-attn\", which is after the encoder/decoder and before the attention module. Can also be \"input\"")
		self.add_argument("--image", default="rgb", type=str, help="What image, if any to use. Possible values are \"rgb\", \"noise\", and \"none\".")
		

	def convert_arg_line_to_args(self, arg_line):
		for arg in arg_line.split():
			if not arg.strip():
				continue
			yield str(arg)

if __name__ == "__main__":
	# Some hacky testing code.
	parser = CommonArgParser(description="Test_text")
	#parser = argparse.ArgumentParser(description="Override_base_method_Text")
	#parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
	#if sys.argv.__len__() == 2:
	#	arg_filename_with_prefix = '@' + sys.argv[1]
	#	args = parser.parse_args([arg_filename_with_prefix])
	#else:
	args = parser.parse_args()
	breakpoint()
