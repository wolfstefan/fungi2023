_base_ = [
    '../_base_/models/swin_transformer_v2/base_384_aug.py', '../_base_/datasets/fungi_bs16_swin_384_class-balanced.py',
    '../_base_/schedules/fungi_bs64_adamw_swin.py', '../_base_/default_runtime.py'
]

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-base-w12_3rdparty_in21k-192px_20220803-f7dc9763.pth'  # noqa
model = dict(
    backbone=dict(
        window_size=[12, 12, 12, 6],
        pretrained_window_sizes=[12, 12, 12, 6],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone',
        )),
    head=dict(num_classes=1604),
    train_cfg=dict(_delete_=True),
)

train_dataloader = dict(
    batch_size=32,
    num_workers=9)

val_dataloader = dict(
    batch_size=64,
    num_workers=9)

train_cfg = dict(max_epochs=6)

optim_wrapper = dict(type='AmpOptimWrapper')

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        end=2100),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=0, by_epoch=False, begin=2100)
]

custom_imports = dict(imports=['mmpretrain_custom'], allow_failed_imports=False)
