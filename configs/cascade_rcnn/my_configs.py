# 新的配置来自基础的配置以更好地说明需要修改的地方
_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'

# 1. 数据集设定
dataset_type = 'CocoDataset'
classes = ['tar']
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/media/lz/lz2/coco/annotations/captions_train2014.json',
        img_prefix='/media/lz/lz2/coco/train2017'),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/media/lz/lz2/coco/annotations/captions_val2014.json',
        img_prefix='/media/lz/lz2/coco/val2017'),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/media/lz/lz2/coco/annotations/captions_test2014.json',
        img_prefix='/media/lz/lz2/coco/test2017'))

# 2. 模型设置

# 将所有的 `num_classes` 默认值修改为5（原来为80）
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=1)],
    # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
))