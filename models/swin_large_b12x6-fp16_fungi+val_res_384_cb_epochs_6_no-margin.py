model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='large',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        drop_path_rate=0.5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1604,
        in_channels=1536,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict())
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
dataset_type = 'Fungi'
data_preprocessor = dict(
    num_classes=1604,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
train_pipeline = [
    dict(type='LoadImageFromFileFungi'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFileFungi'),
    dict(
        type='ResizeEdge',
        scale=438,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=14,
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset=dict(
            type='Fungi',
            data_root='/scratch/slurm_tmpdir/job_22252118/',
            ann_file='FungiCLEF2023_train_metadata_PRODUCTION.csv',
            data_prefix='DF20/',
            pipeline=[
                dict(type='LoadImageFromFileFungi'),
                dict(
                    type='RandomResizedCrop',
                    scale=384,
                    backend='pillow',
                    interpolation='bicubic'),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(
                    type='RandAugment',
                    policies='timm_increasing',
                    num_policies=2,
                    total_level=10,
                    magnitude_level=9,
                    magnitude_std=0.5,
                    hparams=dict(
                        pad_val=[104, 116, 124], interpolation='bicubic')),
                dict(
                    type='RandomErasing',
                    erase_prob=0.25,
                    mode='rand',
                    min_area_ratio=0.02,
                    max_area_ratio=0.3333333333333333,
                    fill_color=[103.53, 116.28, 123.675],
                    fill_std=[57.375, 57.12, 58.395]),
                dict(type='PackInputs')
            ])),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type='Fungi',
        data_root='/scratch/slurm_tmpdir/job_22252118/',
        ann_file='FungiCLEF2023_val_metadata_PRODUCTION.csv',
        data_prefix='DF21/',
        pipeline=[
            dict(type='LoadImageFromFileFungi'),
            dict(
                type='RandomResizedCrop',
                scale=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies='timm_increasing',
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(
    type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'])
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='FungiTest',
        data_root='data/fungi2023/',
        ann_file='FungiCLEF2023_public_test_metadata_PRODUCTION.csv',
        data_prefix='DF21/',
        pipeline=[
            dict(type='LoadImageFromFileFungi'),
            dict(
                type='ResizeEdge',
                scale=384,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=384),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(
    type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'])
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=6.25e-05,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })),
    clip_grad=dict(max_norm=5),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, end=4200),
    dict(type='CosineAnnealingLR', eta_min=0, by_epoch=False, begin=4200)
]
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=64, enable=True)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth'
custom_imports = dict(
    imports=['mmpretrain_custom'], allow_failed_imports=False)
launcher = 'pytorch'
work_dir = './work_dirs/swin_base_b32x4-fp16_fungi+val_res_384_cb_epochs_6'
