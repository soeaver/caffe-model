
### Object Detection Performance on PASCAL VOC.
**1. Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|mAP@50|train speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet101| 78.25 | -- | -- | -- | --
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | -- | --
 resnet152-v2| -- | 2.8 img/s | 9,315MB | -- | --
 wrn50_2| 76.8 | -- | -- | -- | --
 resnext50-32x4d| 77.99 | 3.6 img/s | 5,315MB | -- | --
 resnext101-32x4d| 79.98 | 2.7 img/s | 7,836MB | -- | --
 inception-v3| 78.6 | 4.1 img/s | 4,325MB | -- | --
 inception-v4| 81.49 | 2.6 img/s | 6,759MB | -- | --
 inception-resnet-v2| 80.0 | 2.0 img/s (batch=112) | 11,497MB | -- | --
 densenet-169| -- | -- | -- | -- | --
 densenet-201| 77.7 | 3.9 img/s (batch=72) | 10,073MB | -- | --
 resnet38a| 80.1 | 1.4 img/s | 8,723MB | -- | --
 
 - Performanc, speed and memory are calculated on py-R-FCN-multiGPU (this reproduction) with Nvidia Titan pascal, we do not guarantee that the results can be reproduced under any other conditions;
 - All the models are trained on a single scale (600*1000) with image flipping and train-batch=128 for 80,000 iterations, tested on the same single scale with test-batch=300 and nms=0.3;
 
 
 **2. Comparisons on VOC 2007 test using faster rcnn with inception-v4.**
 
 Method|mAP@50| improvment |test speed
 :---|:---:|:---:|:---:
 baseline inception-v4 | 81.49 | -- | --
 &nbsp;+multi-scale training | -- | -- | --
 &nbsp;+box voting | -- | -- | --
 &nbsp;+image flipping test | -- | -- | --
 &nbsp;+multi-scale testing | -- | -- | --

