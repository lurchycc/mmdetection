from mmdet.apis import init_detector, inference_detector
import mmcv

#模型配置文件和checkpoint文件路径
config_file = '/home/lz/mmdetection-master/configs/queryinst/queryinst_r50_fpn_1x_coco.py'
checkpoint_file = '/home/lz/mmdetection-master/checkpoints/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth'

#初始化model
model = init_detector(config_file,checkpoint_file,device='cuda:0')

# 测试单张图片并展示结果
img = '/home/lz/mmdetection-master/demo/demo.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化
model.show_result(img, result)
# 或者将可视化结果保存为图片
model.show_result(img, result, out_file='result.jpg')

# 测试视频并展示结果
video = mmcv.VideoReader('/home/lz/mmdetection-master/demo/demo.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)