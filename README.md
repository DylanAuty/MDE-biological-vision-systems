# Monocular depth estimation using cues inspired by biological vision systems

This is the official implementation of the paper "Monocular depth estimation using cues inspired by biological vision systems", accepted for publication at the International Conference on Pattern Recognition (ICPR) 2022.

It is a fork of the [original AdaBins repository](https://github.com/shariqfarooq123/AdaBins). The original readme for that repository is included at the end of this ReadMe, for completeness.

## Overview
This repo is comprised of several parts:
1. The instance segmentation module,
2. The semantic segmentation module,
3. The depth estimation module.

The instance and semantic segmentation modules are off-the-shelf models. In some cases, these have been modified to work with the datasets required, and so the versions of them used in this work have been included here for reference.

Instance and semantic segmentation happen offline (although with some modification an end-to-end inference pipeline may be set up, we have not done this due to computational constraints).

The best model in our work uses semantic segmentation from HRNetV2 pretrained on ADE20K, instance segmentation from Cascade Mask-RCNN with Swin-B backbone, pretrained on ADE20K, and the AdaBins pipeline with an EfficientNet-B1 instead of EfficientNet-B5 as the backbone.

The basic steps to get the best model running are:
1. Get NYUD2 and ADE20K datasets,
2. Train the Cascade Mask-RCNN w/ Swin-B backbone on ADE20K,
3. Run instance segmentation inference with this on NYUD2,
4. Acquire HRNetV2 ADE20K checkpoint,
5. Run semantic segmentation inference on NYUD2 using the HRNetV2 checkpoint.
6. Train the depth estimation module using the parameters detailed.

For other models that use different instance or semantic segmentation pipelines, a similar offline training/inference process must be used.

## Prerequisites
### NYUD2
Running training or evaluation requires the NYU Depth V2 Dataset by Silberman et al. (Indoor Segmentation and Support Inference from RGBD Images, ECCV 2012). The specific version used for this work was acquired using the instructions from [the official implementation of BTS (Lee et al. 2019)](https://github.com/cogaplex-bts/bts/tree/master/pytorch). This version of the dataset has been converted to numpy format, and is in the folder structure that this repository requires.

The downloaded dataset should consist of a folder named `nyu`, containing two folders: `official_splits` and `sync`.

### ADE20K
This model requires the ADE20K Places Challenge subset, in order to perform training of some of the instance and semantic segmentation models used.

1. [Download the images](http://sceneparsing.csail.mit.edu/data/ChallengeData2017/images.tar)
2. [Download the annotations](http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar)
3. Uncompress `images.tar` and `annotations_instance.tar` into a common folder, giving:
	```
	/path/to/ADE20K
		+-- annotations_instance
		\-- images
	```
4. Clone the [Places Challenge Toolkit Repo](https://github.com/CSAILVision/placeschallenge)
5. Run `instancesegmentation/evaluation/convert_anns_to_json_dataset.py` on the `training` and `validation` folders inside the `annotations_instance` folder, changing the output name to match. Place the resulting `instance_training_<splitname>_gts.json` files inside the `annotations_instance` folder.

#### Training Swin for instance segmentation on ADE20K
We provide a modified version of the [official Swin repository](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), in the folder `Swin-Transformer-Object-Detection`. The key difference is the inclusion of new config files to allow training on ADE20K.

To train the Swin-B backbone Cascade Mask-RCNN on the ADE20K Places Challenge subset:
1. Fetch the required pretraining checkpoint, `swin_base_patch4_window12_384_22kto1k.pth`, from [this URL](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth).
2. Configure the Swin environment (we used Docker) accordinging to the instructions in the README in that folder.
3. Inside the Swin folder, open `./configs/_base_/datasets/ade20k_instance.py` in a text editor.
4. Modify:
	- Line 2, change `data_root` on to point to the ADE20K folder,
	- Lines 38, 44, and 50 to point to the training, validation, and validation (again) ground-truth JSON files. If you placed them according to step 5 in the ADE20K section above, these shouldn't need modification.
5. Run:
	```
	tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k.py 1 --cfg-options model.pretrained=checkpoints/swin_base_patch4_window12_384_22kto1k.pth
	```
	Note that something causes RAM usage to consistently increase every time evaluation is run. In the interests of transparency, we include the aptly named `keep_resuming_until_success.sh` that restarts training if something causes it to fail.

#### Swin inference on NYUD2
This is an instance segmentation model, outputting semantic labels and per-pixel areas-of-instance-of-this-pixel. Running this section generates the `instance_areas_ade20k_swin_<number>.npz` and `instance_labels_ade20k_swin_<number>.npz` files corresponding to the `rgb_<number>.jpg` and `sync_depth_<number>.png` files already present in the dataset.

1. `cd` to the Swin folder in this repository
2. Run the `tools/nyud2_inference.py` script, using the following command: 
	```
	python tools/nyud2_inference.py --config configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k.py --checkpoint work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_ade20k/epoch_36.pth --images data/nyu
	```
	Modify the path under `--images` to point to your NYUD2 folder.

#### HRNetV2 inference on NYUD2
This is a semantic segmentation only model. Running this section generates the `semantic_seg_<number>.npy` files corresponding to the `rgb_<number>.jpg` and `sync_depth_<number>.png` files already present in the dataset.

1. `cd` into the `semantic-segmentation-pytorch` folder, which is a fork of the [Official ADE20K semantic segmentation toolkit](https://github.com/CSAILVision/semantic-segmentation-pytorch)
3. Download the HRNetV2 checkpoints from [this link](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/), and place them in the folder `ckpt/ade20k-hrnetv2-c1`.
4. Follow any setup/installation instructions in the README in the `semantic-segmentation-pytorch` folder.
5. Modify `demo_test.sh` on line 5 to point to your NYUD2 folder. This script will search the directory for images, run inference on them, and save the results back to the same directory with different filenames.

## Evaluation
Evaluation is performed using `evaluate.py /path/to/params/file.txt`. To do so, datasets must be set up as outline in the prerequisites section.

A base parameter file for evaluation, `params/args_test_nyu_BASE.txt`, is included. Checkpoints are not provided with supplementary material due to file size limit.

First, modify the following arguments to point to the `nyu/sync/` and `nyu/official_splits/test/` directories respectively:

1. `--data_path` and `--gt_path` should both have the absolute path of `nyu/sync/`
2. `--data_path_eval` and `gt_path_eval` should both have the absolute path of `nyu/official_splits/test/`.

To run evaluation on a checkpoint, ensure that the following arguments match in both the evaluation parameter file, and in the training parameter file for the run that produced the checkpoint to be used:

* `--use_semantics`
* `--use_instance_segmentation`
* `--encoder_name`
* `--insertion_point`
If one of these is not present in the training params file, it should not be present in the evaluation file either.

You must also ensure that `--checkpoint_path` points to the checkpoint you wish to run evaluation on (and the checkpoint must have been produced with training parameters matching the above list).

All other parameters should match those in any of the provided evaluation parameter files.


## Training
Training is performed using `train.py /path/to/training/params.txt`. Training parameters as used for our experiments are provided in the `params` folder. We note that the system used for this work comprised 2x NVIDIA GeForce GTX 1080 graphics cards, with 8GB VRAM each. 

To reproduce an experiment, modify the following arguments to point to the `nyu/sync/` and `nyu/official_splits/test/` directories respectively:

1. `--data_path` and `--gt_path` should both have the absolute path of `nyu/sync/`
2. `--data_path_eval` and `gt_path_eval` should both have the absolute path of `nyu/official_splits/test/`.

Also, modify the `--root` argument to point to the folder in which the experiment folder will be created. The experiment folder that is created will also contain the tensorboard run files, so setting your tensorboard `logdir` to match `--root` works well.

All other arguments should be left the same, and training should be performed on a system with similar hardware capabilities.

If running a new experiment that has not been run before, ensure that you modify the `--name` attribute, or you may end up with a previous experiment being overwritten.


## Experiment Naming Convention
All experiment names (parameter file names, the `--name` parameter in training parameters, and therefore the generated checkpoint names) detail the values for the following train/eval parameters:

* `--dataset`: Always `nyu`
* `--encoder_name`: Always `efficientnet-b1`, but can be `efficientnet-b5` if sufficient compute is available
* `--use_semantics` (sometimes prefaced with `sem_` in param filenames):
	* `glove-25d` uses results from running inference on NYUD2 with HRNetV2 trained on ADE20K, embedded with 25 dimensional GloVe embeddings.
	* `glove-25d-ade20k-places` uses only the semantic labels generated by the Cascade Mask-RCNN with Swin-B backbone trained on the ADE20K Places Challenge subset. Encoded as `glove-25d`.
	* `glove-25d-ade20k-places-human-sizes` is as the previous, except that per-class absolute dimensions in metres are provided to the network. These are embedded as described in the paper.
* `--use_instance_segmentation` (sometimes prefaced with `inst_` in param file names):
	* `coco` uses instance segmentation and mask areas from using Mask-RCNN trained on MSCOCO to run inference on NYUD2. Class label names embedded as in `glove-25d` semantics.
	* `ade20k_swin` uses labels and mask areas from the Cascade Mask-RCNN w/ Swin-B backbone trained on the ADE20K Places Challenge subset, run on NYUD2.
	* `ade20k_swin_human_sizes` is as the previous, except include absolute per-label dimensions in metres. Full details in the paper.
	* `ade20k_swin_bbox` Is the same as `ade20k_swin` but uses bounding box areas instead of mask areas for the instance areas.
	* `ade20k_swin_bbox_human_sizes` is the same as `ade20k_swin_human_sizes` but uses bounding box instead of mask instance areas.
* `--insertion_point`
	* Always `input` (can also be `before-attn` to attach all added information after the encoder/decoder but before the AdaBins module, but this performs worse in all cases.)

Our parameter file names match our checkpoint names.


# AdaBins (Original README, unedited)

Official implementation of [Adabins: Depth Estimation using adaptive bins](https://arxiv.org/abs/2011.14141)
## Download links
* You can download the pretrained models "AdaBins_nyu.pt" and "AdaBins_kitti.pt" from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
* You can download the predicted depths in 16-bit format for NYU-Depth-v2 official test set and KITTI Eigen split test set [here](https://drive.google.com/drive/folders/1b3nfm8lqrvUjtYGmsqA5gptNQ8vPlzzS?usp=sharing)

## Inference
Move the downloaded weights to a directory of your choice (we will use "./pretrained/" here). You can then use the pretrained models like so:

```python
from models import UnetAdaptiveBins
import model_io
from PIL import Image

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

# NYU
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
pretrained_path = "./pretrained/AdaBins_nyu.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)

# KITTI
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "./pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)
```
Note that the model returns bin-edges (instead of bin-centers).

**Recommended way:** `InferenceHelper` class in `infer.py` provides an easy interface for inference and handles various types of inputs (with any prepocessing required). It uses Test-Time-Augmentation (H-Flips) and also calculates bin-centers for you:
```python
from infer import InferenceHelper

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a batched rgb tensor
example_rgb_batch = ...  
bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")

```
## TODO:
* Add instructions for Evaluation and Training.
* Add Colab demo
* Add UI demo
* Remove unnecessary dependencies
