
## Object Detection

### We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
we are releasing the training code and files, the models and more experiments will come soon.

### Object Detection Performance on PASCAL VOC.
**1. Original faster rcnn training on VOC 2007+2012 trainval and testing on VOC 2007 test.**

 Network|mAP@50(%)|training<br/>speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 70.02 | 9.5 img/s | 1,235MB | 17.5 img/s | 989MB
 resnet101| -- | -- | -- | -- | --
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | 7.1 img/s | 4,573MB
 resnet152-v2| 80.72 | 2.8 img/s | 9,315MB | 6.2 img/s | 6,021MB
 wrn50_2| 78.59 | 2.1 img/s | 4,895MB | 4.9 img/s | 3,499MB
 resnext50-32x4d| 77.99 | 3.6 img/s | 5,315MB | 7.4 img/s | 4,305MB
 resnext101-32x4d| 79.98 | 2.7 img/s | 7,836MB | 6.3 img/s | 5,705MB
 resnext101-64x4d| 80.71 | 2.0 img/s<br/> (batch=96) | 11,277MB | 3.7 img/s | 9,461MB
 inception-v3| 78.6 | 4.1 img/s | 4,325MB | 7.3 img/s | 3,445MB
 inception-v4| 81.49 | 2.6 img/s | 6,759MB | 5.4 img/s | 4,683MB
 inception-resnet-v2| 80.0 | 2.0 img/s<br/> (batch=112) | 11,497MB | 3.2 img/s | 8,409MB
 densenet-161| -- | -- | -- | -- | --
 densenet-201| 77.53 | 3.9 img/s<br/> (batch=72) | 10,073MB | 5.5 img/s | 9,955MB
 resnet38a| 80.1 | 1.4 img/s | 8,723MB | 3.4 img/s | 5,501MB
 
 - To reduce memory usage, we merge all the models batchnorm layer parameters into scale layer, more details please refer to [faster-rcnn-resnet](https://github.com/Eniac-Xie/faster-rcnn-resnet#modification) or [pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn/blob/master/tools/gen_merged_model.py);
 - We also split the deploy file to rpn deploy file and rcnn deploy file for adopting more testing tricks.
 - Performanc, speed and memory are calculated on [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) with Nvidia Titan pascal, we do not guarantee that the results can be reproduced under any other conditions;
 - All the models are trained on a single scale (600*1000) with image flipping and train-batch=128 for 80,000 iterations, tested on the same single scale with test-batch=300 and nms=0.3;
 
 
 **2. Comparisons on VOC 2007 test using faster rcnn with inception-v4.**
 
 Method|mAP@50| improvment |test speed
 :---|:---:|:---:|:---:
 baseline inception-v4 | 81.49 | -- | 5.4 img/s
 &nbsp;+multi-scale training | 83.79 | 2.30 | 5.4 img/s
 &nbsp;+box voting | 83.95 | 0.16 | 5.4 img/s
 &nbsp;+nms=0.4 | 84.22 | 0.27 | 5.4 img/s
 &nbsp;+image flipping test | 84.54 | 0.32 | 2.7 img/s
 &nbsp;+multi-scale testing | 85.78 | 1.24 | 0.13 img/s
 
 - The SCALES for multi-scale training is (200, 400, 600, 800, 1000) and MAX_SIZE is 1666; 
 - For multi-scale training, we double the training iterations (160,000 for VOC0712trainval);
 - The SCALES for multi-scale testing is (400, 600, 800, 1000, 1200) and MAX_SIZE is 2000;

