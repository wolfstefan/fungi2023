_base_ = [
    '../_base_/models/swin_transformer/base_384_aug.py', '../_base_/datasets/fungi_bs16_swin_384.py',
    '../_base_/schedules/fungi_bs64_adamw_swin.py', '../_base_/default_runtime.py'
]

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth'  # noqa
model = dict(
    backbone=dict(
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
    num_workers=14)

val_dataloader = dict(
    batch_size=64,
    num_workers=12)

train_cfg = dict(max_epochs=24)

optim_wrapper = dict(type='AmpOptimWrapper')

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        end=4200),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=0, by_epoch=False, begin=4200)
]

custom_imports = dict(imports=['mmpretrain_custom'], allow_failed_imports=False)