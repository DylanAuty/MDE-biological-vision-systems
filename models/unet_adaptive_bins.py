import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT

# _calc_same_pad, conv2d_same and Conv2dSame taken from EfficientNet implementation (https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/conv2d_layers.py)
from typing import Union, List, Tuple, Optional, Callable

def _calc_same_pad(i: int, k: int, s: int, d: int):
	return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def conv2d_same(
		x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
		padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
	ih, iw = x.size()[-2:]
	kh, kw = weight.size()[-2:]
	pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
	pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
	x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
	return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
	""" Tensorflow like 'SAME' convolution wrapper for 2D convolutions
	"""

	# pylint: disable=unused-argument
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(Conv2dSame, self).__init__(
			in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

	def forward(self, x):
		return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleBN(nn.Module):
	def __init__(self, skip_input, output_features):
		super(UpSampleBN, self).__init__()

		self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
								  nn.BatchNorm2d(output_features),
								  nn.LeakyReLU(),
								  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
								  nn.BatchNorm2d(output_features),
								  nn.LeakyReLU())

	def forward(self, x, concat_with):
		up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
		f = torch.cat([up_x, concat_with], dim=1)
		return self._net(f)


class DecoderBN(nn.Module):
	def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, mode="AdaBins"):
		super(DecoderBN, self).__init__()
		features = int(num_features)

		self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

		if num_features == 2048:	# Used for EfficientNet B5
			skip_feat_add = [64, 24, 16, 8]
		elif num_features == 1280:	# Used for EfficientNet B1
			skip_feat_add = [0, 0, 0, 0]
		
		self.up1 = UpSampleBN(skip_input=features // 1 + 112 + skip_feat_add[0], output_features=features // 2)
		self.up2 = UpSampleBN(skip_input=features // 2 + 40 + skip_feat_add[1], output_features=features // 4)
		self.up3 = UpSampleBN(skip_input=features // 4 + 24 + skip_feat_add[2], output_features=features // 8)
		self.up4 = UpSampleBN(skip_input=features // 8 + 16 + skip_feat_add[3], output_features=features // 16)

		#		 self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
		self.mode = mode
		if self.mode == "AdaBins":
			# Output to be used with the AdaBins module.
			self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
		elif self.mode == "noAdaBins":
			# A direct depth output.
			self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)
		# self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

	def forward(self, features):
		x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
			11]

		x_d0 = self.conv2(x_block4)			# in: bottleneck_features, out: features

		x_d1 = self.up1(x_d0, x_block3)		# in: features + 
		x_d2 = self.up2(x_d1, x_block2)
		x_d3 = self.up3(x_d2, x_block1)
		x_d4 = self.up4(x_d3, x_block0)
		#		 x_d5 = self.up5(x_d4, features[0])
		out = self.conv3(x_d4)
		# out = self.act_out(out)
		# if with_features:
		#	 return out, features[-1]
		# elif with_intermediate:
		#	 return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
		return out


class Encoder(nn.Module):
	def __init__(self, backend):
		super(Encoder, self).__init__()
		self.original_model = backend

	def forward(self, x):
		features = [x]
		for k, v in self.original_model._modules.items():
			if (k == 'blocks'):
				for ki, vi in v._modules.items():
					features.append(vi(features[-1]))
			else:
				features.append(v(features[-1]))
		return features


class UnetAdaptiveBins(nn.Module):
	def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear', encoder_name="efficientnet-b5", semantics_mode=None, instance_segmentation_mode=None, insertion_point="before-attn", image="rgb"):
		super(UnetAdaptiveBins, self).__init__()
		self.num_classes = n_bins
		self.min_val = min_val
		self.max_val = max_val
		self.encoder = Encoder(backend)
		self.semantics_mode = semantics_mode
		self.instance_segmentation_mode = instance_segmentation_mode
		self.insertion_point=insertion_point
		self.image = image
		self.encoder_name = encoder_name
		self.image_pre_encode = None

		self.num_decoded_channels = 128		# By default this is 128 but will change with different experiments.

		num_channels_to_add = UnetAdaptiveBins.get_num_channels_to_add(
				encoder_name=self.encoder_name,
				semantics_mode=self.semantics_mode,
				instance_segmentation_mode=self.instance_segmentation_mode,
				image=self.image)

		if self.insertion_point == "before-attn":
			self.num_decoded_channels += num_channels_to_add

		if self.semantics_mode is not None:
			if self.semantics_mode == "glove-25d-inst-areas":
				self.semantics_areas_fc = nn.Sequential(
						nn.Conv2d(1, 10, kernel_size=1),
						nn.ReLU(),
						nn.Conv2d(10, 10, kernel_size=1),
						nn.ReLU()
						)
			if "human-sizes" in self.semantics_mode:
				self.semantics_absolute_sizes_fc = nn.Sequential(
						nn.Conv2d(3, 10, kernel_size=1),
						nn.ReLU(),
						nn.Conv2d(10, 10, kernel_size=1),
						nn.ReLU()
						)

		if self.instance_segmentation_mode is not None:
			self.instance_areas_fc = nn.Sequential(
					nn.Conv2d(1, 10, kernel_size=1),
					nn.ReLU(),
					nn.Conv2d(10, 10, kernel_size=1),
					nn.ReLU()
					)

			if "human_sizes" in self.instance_segmentation_mode:
				self.instance_absolute_sizes_fc = nn.Sequential(
					nn.Conv2d(3, 10, kernel_size=1),
					nn.ReLU(),
					nn.Conv2d(10, 10, kernel_size=1),
					nn.ReLU()
					)

		if "noAdaBins" not in self.encoder_name:
			self.adaptive_bins_layer = mViT(self.num_decoded_channels, n_query_channels=128, patch_size=16,
											dim_out=n_bins,
											embedding_dim=128, norm=norm)

		if "efficientnet-b5" in self.encoder_name:
			self.decoder = DecoderBN(num_classes=128, num_features=2048, bottleneck_features=2048)
		elif "efficientnet-b1" in self.encoder_name:
			if "noAdaBins" in self.encoder_name:
				self.decoder = DecoderBN(num_classes=128, num_features=1280, bottleneck_features=1280, mode="noAdaBins")
			else:
				self.decoder = DecoderBN(num_classes=128, num_features=1280, bottleneck_features=1280)

		if "noAdaBins" not in self.encoder_name:
			self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
										  nn.Softmax(dim=1))

	def forward(self, x, semantics=None, instance_labels=None, instance_areas=None, **kwargs):
		if self.insertion_point == "input":
			if semantics is not None:
				if self.semantics_mode == "glove-25d-inst-areas":
					# In this case, semantics label dims 0-24 are glove embedding of label and 
					# 25 is proportion of image taken up by class of that pixel.
					x = torch.cat((x, semantics[:, 0:25, :, :].float()), dim=1)
					semantics_areas = semantics[:, 25:26, :, :].float()
					semantics_areas = self.semantics_areas_fc(semantics_areas)
					x = torch.cat((x, semantics_areas), dim=1)
				elif "human-sizes" in self.semantics_mode:
					# In this case, semantics label dims 0-N are N-dimensional word embedding of label and 
					# -3:(last) are absolute dimensions of class
					x = torch.cat((x, semantics[:, 0:-3, :, :].float()), dim=1)
					semantics_abs_sizes = semantics[:, -3:, :, :].float()
					semantics_abs_sizes = self.semantics_absolute_sizes_fc(semantics_abs_sizes)
					x = torch.cat((x, semantics_abs_sizes), dim=1)
				else:
					x = torch.cat((x, semantics.float()), dim=1)
			if instance_labels is not None:
				x = torch.cat((x, instance_labels.float()), dim=1)
			if instance_areas is not None:
				if "human_sizes" in self.instance_segmentation_mode:
					# In this case, instance_areas channel 1 is actual instance areas, and instance_areas channel 2 is 
					# absolute size. This is because it's a lot simpler to misuse one of these inputs than it is to add 
					# a new one.
					instance_areas_1 = instance_areas[:, 0:1, :, :] / (x.shape[2] * x.shape[3])
					instance_areas_1 = self.instance_areas_fc(instance_areas_1)
					x = torch.cat((x, instance_areas_1), dim=1)
					instance_abs_sizes = instance_areas[:, 1:4, :, :]
					instance_abs_sizes = self.instance_absolute_sizes_fc(instance_abs_sizes)
					x = torch.cat((x, instance_abs_sizes), dim=1)
				else:
					instance_areas = instance_areas / (x.shape[2] * x.shape[3])	# Normalise areas to be fraction of image size
					instance_areas = self.instance_areas_fc(instance_areas)
					x = torch.cat((x, instance_areas), dim=1)

		# If using no image, check that there's going to be something left, then nuke the first 3 channels.
		if self.image == "none":
			if x.shape[1] <= 3:
				sys.exit("Error: Add more auxiliary information at input if using no image")
			else:
				x = x[:, 3:, :, :]

		encoded = self.encoder(x)
		unet_out = self.decoder(encoded, **kwargs)

		if "noAdaBins" in self.encoder_name:
			# This means no AdaBins module, so decoder should be outputting one channel (depth) only.
			return None, F.relu(unet_out) + 0.0001	# unet_out is the prediction. None is where the bin_edges would be in the train loop. The relu and offset are to allow the SILog loss to work.
		else:
			if self.insertion_point == "before-attn":
				# If semantics are present, downsample to match unet_out and concat
				if semantics is not None:
					semantics = F.interpolate(semantics, size=(unet_out.shape[2], unet_out.shape[3]), mode='nearest').float()
					if self.semantics_mode == "glove-25d-inst-areas":
						unet_out = torch.cat((unet_out, semantics[:, 0:25, :, :]), dim=1)
						semantics_areas = semantics[:, 25:26, :, :].float()
						semantics_areas = self.semantics_areas_fc(semantics_areas)
						unet_out = torch.cat((unet_out, semantics_areas), dim=1)
					elif "human-sizes" in self.semantics_mode:
						# In this case, semantics label dims 0-N are N-dimensional word embedding of label and 
						# -3:(last) are absolute dimensions of class
						x = torch.cat((x, semantics[:, 0:-3, :, :].float()), dim=1)
						semantics_abs_sizes = semantics[:, -3:, :, :].float()
						semantics_abs_sizes = self.semantics_absolute_sizes_fc(semantics_abs_sizes)
						x = torch.cat((x, semantics_abs_sizes), dim=1)
					else:
						unet_out = torch.cat((unet_out, semantics), dim=1)
				# If instance labels are present, downsample to match unet_out and concat
				if instance_labels is not None:
					instance_labels = F.interpolate(instance_labels, size=(unet_out.shape[2], unet_out.shape[3]), mode='nearest').float()
					unet_out = torch.cat((unet_out, instance_labels), dim=1)
				# The same thing with instance areas, but they'll get put through some other layers before concatenation.
				if instance_areas is not None:
					instance_areas = F.interpolate(instance_areas, size=(unet_out.shape[2], unet_out.shape[3]), mode='nearest').float()
					if "human_sizes" in self.instance_segmentation_mode:
						# In this case, instance_areas channel 1 is actual instance areas, and instance_areas channel 2 is 
						# absolute size. This is because it's a lot simpler to misuse one of these inputs than it is to add 
						# a new one.
						instance_areas_1 = instance_areas[:, 0:1, :, :] / (x.shape[2] * x.shape[3])
						instance_areas_1 = self.instance_areas_fc(instance_areas_1)
						unet_out = torch.cat((unet_out, instance_areas_1), dim=1)
						instance_abs_sizes = instance_areas[:, 1:4, :, :]
						instance_abs_sizes = self.instance_absolute_sizes_fc(instance_abs_sizes)
						unet_out = torch.cat((unet_out, instance_abs_sizes), dim=1)
					else:
						instance_areas = instance_areas / (x.shape[2] * x.shape[3])	# Normalise areas to be fraction of image size
						instance_areas = self.instance_areas_fc(instance_areas)
						unet_out = torch.cat((unet_out, instance_areas), dim=1)


			bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
			out = self.conv_out(range_attention_maps)

			# Post process
			# n, c, h, w = out.shape
			# hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

			bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
			bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
			bin_edges = torch.cumsum(bin_widths, dim=1)

			centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
			n, dout = centers.size()
			centers = centers.view(n, dout, 1, 1)

			pred = torch.sum(out * centers, dim=1, keepdim=True)

			return bin_edges, pred

	def get_1x_lr_params(self):  # lr/10 learning rate
		return self.encoder.parameters()

	def get_10x_lr_params(self):  # lr learning rate
		if "noAdaBins" in self.encoder_name:
			modules = [self.decoder]
		else:
			modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
		for m in modules:
			yield from m.parameters()

	@classmethod
	def build(cls, n_bins, encoder_name="efficientnet-b5", insertion_point="before-attn", **kwargs):
		# First, set up the base model that the rest of the network will be built around.
		if "efficientnet-b5" in encoder_name:
			basemodel_name = 'tf_efficientnet_b5_ap'
		elif "efficientnet-b1" in encoder_name:
			basemodel_name = 'tf_efficientnet_b1_ap'

		print(f'Loading base model ({basemodel_name})...', end='')
		basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
		print('Done.')

		# Remove last layer
		print('Removing last two layers (global_pool & classifier).')
		basemodel.global_pool = nn.Identity()
		basemodel.classifier = nn.Identity()

		# If insertion point for semantics and instance stuff is the start, modify the encoder accordingly
		if insertion_point == "input":
			print("Extending channels of first layer (args.insertion_point == input)")
			
			num_channels_to_add = UnetAdaptiveBins.get_num_channels_to_add(
					encoder_name=encoder_name,
					semantics_mode=kwargs['semantics_mode'],
					instance_segmentation_mode=kwargs['instance_segmentation_mode'],
					image=kwargs['image']
					)

			# Strip off first layer, replace with a bigger one, but try to keep the first three channel's worth of weights
			conv_stem_weights = basemodel.conv_stem.weight.clone()
			basemodel.conv_stem = Conv2dSame((3 + num_channels_to_add), 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
			with torch.no_grad():
				basemodel.conv_stem.weight[:, 0:3, :, :] = conv_stem_weights

			# If the image flag is "none" (the string, not None), we're also going to strip off the RGB channels (i.e. remake it).
			if 'image' in kwargs.keys():
				if kwargs['image'] == "none":
					if num_channels_to_add < 1:
						sys.exit("Too few input channels - add more inputs")
					basemodel.conv_stem = Conv2dSame((num_channels_to_add), 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

		# Building Encoder-Decoder model
		print('Building Encoder-Decoder model..', end='')
		m = cls(basemodel, n_bins=n_bins, encoder_name=encoder_name, insertion_point=insertion_point, **kwargs)
		print('Done.')
		return m

	
	@staticmethod
	def get_num_channels_to_add(encoder_name, semantics_mode, instance_segmentation_mode, image):
		"""
		Method to work out how many channels to tack on to wherever, depending on the semantics and instance segmentation info
		that will be used.
		"""
		num_channels_to_add = 0
		if semantics_mode is not None:
			if "raw" in semantics_mode:
				num_channels_to_add += 1
			elif semantics_mode == "glove":
				num_channels_to_add += 300
			elif "glove-25d" in semantics_mode:
				num_channels_to_add += 25
			else:
				sys.exit("Error [models/unet_adaptive_bins.py]: semantics mode not recognised")

			if "inst-areas" in semantics_mode:
				num_channels_to_add += 10
			if "human-sizes" in semantics_mode:
				num_channels_to_add += 10

		if instance_segmentation_mode is not None:
			if instance_segmentation_mode == "raw":
				num_channels_to_add += 1
			elif instance_segmentation_mode == "coco" or "ade20k_swin" in instance_segmentation_mode:
				num_channels_to_add += 35	# 25 channels for label embedding, 10 for size.
				# Instance areas are normalised then run through the below.
			if "human_sizes" in instance_segmentation_mode:
				num_channels_to_add += 10
				# Instance areas (not absolute areas) are normalised in the fwd pass then run through the below.

		return num_channels_to_add


if __name__ == '__main__':
	model = UnetAdaptiveBins.build(100)
	x = torch.rand(2, 3, 480, 640)
	bins, pred = model(x)
	print(bins.shape, pred.shape)
