# GraphBuilder_NYUD2.py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import colorcet as cc
import numpy as np
import torch

class GraphBuilder_NYUD2():
	"""A class for setting up a grid of images relating to depth experiments.
	Purpose is to centralise all code for handling ranges, colour schemes, etc, 
	so that consistent figures are generated.
	Use by calling add_image() with the batched rgbs, depthOuts, etc., and then
	access the figure and do whatever with it.
	When finished, clear the figure by calling the reset() method.

	There are several different versions of the GraphBuilder class. This one is
	to be used to produce an eval output for the NYUv2-OC++ dataset.
	"""

	def __init__(self, num_samples, semantics=False, instance_labels=False):
		self.semantics = semantics
		self.instance_labels = instance_labels
		self.num_samples = num_samples
		self.num_columns = 3
		if semantics:
			self.num_columns += 1
		if instance_labels:
			self.num_columns += 1
		plt.axis('off')
		width = self.num_columns * (7/3)
		height = self.num_samples * width / self.num_columns * (0.75) + 0.3	# 0.75 is the height:width ratio for NYUD2
		self.fig, self.ax = plt.subplots(self.num_samples, self.num_columns, figsize=(width, height))
		self.curr_sample = 0
		self.ax[self.curr_sample, 0].set_title('RGB')
		self.ax[self.curr_sample, 1].set_title('G.T. Depth')
		self.ax[self.curr_sample, 2].set_title('Pred. Depth')
		curr_idx = 2
		if semantics:
			curr_idx += 1
			self.ax[self.curr_sample, curr_idx].set_title('Input Semantics')
		if instance_labels:
			curr_idx += 1
			self.ax[self.curr_sample, curr_idx].set_title('Input Instance Classes')
		
		[axi.set_axis_off() for axi in self.ax.ravel()]

	
	def add_image(self, rgbs, gtDepth, depthOuts, semantics=None, instance_labels=None):
		"""Method to be called to set up a plot of self.num_samples samples, or fewer.
		All samples will be from the same batch, and all inputs are batched (BxCxHxW).
		Once this has set the figure up, it can be written to tensorboard or similar.
		"""
		if self.curr_sample > self.num_samples:
			return
		else:
			# Add as many samples as possible from the batch, up to the max number
			# if there's fewer samples in the batch, then do fewer
			samples_to_plot = 0
			if self.num_samples < rgbs.shape[0]:
				samples_to_plot = self.num_samples
			else:
				samples_to_plot = rgbs.shape[0]
			
			for self.curr_sample in range(0, samples_to_plot):
				# The image can be safely added to the plot.
				# RGBs
				rgb_min = torch.min(rgbs[self.curr_sample, :])
				rgb_max = torch.max(rgbs[self.curr_sample, :])
				rgbs[self.curr_sample, :] = (rgbs[self.curr_sample, :] - rgb_min) / (rgb_max - rgb_min)
				self.ax[self.curr_sample, 0].imshow(np.transpose(rgbs[self.curr_sample, :].detach().cpu().numpy(), (1, 2, 0)))
				self.ax[self.curr_sample, 0].axis('off')
				# GT Depth
				vmin = torch.min(gtDepth[self.curr_sample, 0])
				vmax = torch.max(gtDepth[self.curr_sample, 0])
				self.ax[self.curr_sample, 1].imshow(gtDepth[self.curr_sample, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap='inferno_r')
				self.ax[self.curr_sample, 1].axis('off')

				# Predicted Depth
				# Uses vmin and vmax from GTDepth so they're comparable
				self.ax[self.curr_sample, 2].imshow(depthOuts[self.curr_sample, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap='inferno_r')
				self.ax[self.curr_sample, 2].axis('off')
				
				curr_idx = 2

				if semantics is not None:
					curr_idx += 1
					self.ax[self.curr_sample, curr_idx].imshow(semantics[self.curr_sample, 0])
					self.ax[self.curr_sample, curr_idx].axis('off')
				
				if instance_labels is not None:
					curr_idx += 1
					self.ax[self.curr_sample, curr_idx].imshow(instance_labels[self.curr_sample, 0])
					self.ax[self.curr_sample, curr_idx].axis('off')

			# Plot is populated now, so fiddle with the layout a bit and set up the DPI
			self.fig.tight_layout()
			self.fig.subplots_adjust(hspace=0.02, wspace=0.04)
			self.fig.dpi = 200


	def reset(self):
		"""A convenience function. After setting up with self.add_image() and plotting with
		tensorboard or similar, self.reset() should be called in order to clear the figure and reinitialise the
		counter self.curr_sample. If this isn't reset, it will refuse to plot (to avoid filling the log with
		pointless figures).
		"""
		self.__init__(self.num_samples, self.semantics, self.instance_labels)
