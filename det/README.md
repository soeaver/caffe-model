
### Object Detection Performance on PASCAL VOC.
**1. Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|mAP@50|train speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 70.02 | 9.5 img/s | 1,235MB | -- | --
 resnet101| 78.25 | -- | -- | -- | --
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | -- | --
 resnet152-v2| 80.72 | 2.8 img/s | 9,315MB | -- | --
 wrn50_2| 78.59 | 2.1 img/s | 4,895MB | -- | --
 resnext50-32x4d| 77.99 | 3.6 img/s | 5,315MB | -- | --
 resnext101-32x4d| 79.98 | 2.7 img/s | 7,836MB | -- | --
 resnext101-64x4d| -- | 2.0 img/s (batch=96) | 11,277MB | -- | --
 inception-v3| 78.6 | 4.1 img/s | 4,325MB | -- | --
 inception-v4| 81.49 | 2.6 img/s | 6,759MB | -- | --
 inception-resnet-v2| 80.0 | 2.0 img/s (batch=112) | 11,497MB | -- | --
 densenet-161| -- | -- | -- | -- | --
 densenet-201| 77.53 | 3.9 img/s (batch=72) | 10,073MB | -- | --
 resnet38a| 80.1 | 1.4 img/s | 8,723MB | -- | --
 
 - Performanc, speed and memory are calculated on py-R-FCN-multiGPU (this reproduction) with Nvidia Titan pascal, we do not guarantee that the results can be reproduced under any other conditions;
 - All the models are trained on a single scale (600*1000) with image flipping and train-batch=128 for 80,000 iterations, tested on the same single scale with test-batch=300 and nms=0.3;
 
 
 **2. Comparisons on VOC 2007 test using faster rcnn with inception-v4.**
 
 Method|mAP@50| improvment |test speed
 :---|:---:|:---:|:---:
 baseline inception-v4 | 81.49 | -- | --
 &nbsp;+multi-scale training | 83.79 | 2.30 | --
 &nbsp;+box voting | 83.95 | 0.16 | --
 &nbsp;+nms=0.4 | 84.22 | 0.27 | --
 &nbsp;+image flipping test | 84.54 | 0.32 | --
 &nbsp;+multi-scale testing | 85.45 | 0.91 | --

