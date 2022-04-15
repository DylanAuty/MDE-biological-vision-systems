dataset_type = 'CocoDataset'
data_root = 'data/ADE20K_instseg/'

classes = ['bed', 'windowpane', 'cabinet', 'person', 'door', 'table', 'curtain', 'chair', 'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair', 'seat', 'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sink', 'fireplace', 'refrigerator', 'stairs', 'case', 'pool table', 'pillow', 'screen door', 'bookcase', 'coffee table', 'toilet', 'flower', 'book', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'apparel', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship', 'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag', 'minibike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'plate', 'monitor', 'bulletin board', 'radiator', 'glass', 'clock', 'flag']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_instance/instance_training_gts.json',
        img_prefix=data_root + 'images/training',
		classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_instance/instance_validation_gts.json',
        img_prefix=data_root + 'images/validation',
		classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_instance/instance_validation_gts.json',
        img_prefix=data_root + 'images/validation',
		classes=classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
