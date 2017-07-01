
### Object Detection Performance on PASCAL VOC.
**1. Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|mAP@50|train speed|train memory cost|test speed|test memory cost
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet101| 78.25 | -- | -- | -- | --
 resnet101-v2| -- | 3.1 img/s | 6,495MB | -- | --
 resnet152-v2| -- | -- | -- | -- | --
 wrn50_2| 76.8 | -- | -- | -- | --
 resnext50-32x4d| -- | -- | -- | -- | --
 resnext101-32x4d| 79.94 | -- | -- | -- | --
 inception-v3| -- | -- | -- | -- | --
 inception-v4| -- | -- | -- | -- | --
 inception-resnet-v2| **80.9** | -- | -- | -- | --
 densenet-169| -- | -- | -- | -- | --
 densenet-201| 77.7 | -- | -- | -- | --
 resnet38a| 80.1 | -- | -- | -- | --
 
 - All the models are trained on single scale (600*1000), and tested on the same single scale;
 
