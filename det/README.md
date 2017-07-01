
### Object Detection Performance on PASCAL VOC.
**1. Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|peformance(mAP@50)|test speed (fps)|train mothed|test method
 :---:|:---:|:---:|:---:|:---:
 resnet101| 78.25 | -- | ss | ss
 resnet101-v2| -- | -- | ss | ss
 resnet152-v2| -- | -- | ss | ss
 wrn50_2(resnet50_1x128d)| 76.8 | -- | ss | ss
 resnext50-32x4d| -- | -- | ss | ss
 resnext101-32x4d| 79.94 | -- | ss | ss
 inception-v3| -- | -- | ss | ss
 inception-v4| -- | -- | ss | ss
 inception-resnet-v2| **80.9** | -- | ss | ss
 densenet-169| -- | -- | ss | ss
 densenet-201| 77.7 | -- | ss | ss
 resnet38a| 80.1 | -- | ss | ss
 
