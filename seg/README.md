## Object Segmentation

### We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
we are releasing the training code and files, the models and more experiments will come soon.

### Object Segmentation Performance on PASCAL VOC.
**1. PSPNet training on [SBD](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf) (10,582 images) and testing on VOC 2012 validation (1,449 images).**

 Network|mIoU(%)|pixel acc(%)|train speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 resnet101-v2| 77.94 | 94.94 | -- img/s | --MB | -- img/s | --MB
 resnet101-v2-selu| 77.10 | 94.80 | -- img/s | --MB | -- img/s | --MB
 air101| 77.64 | 94.93 | -- img/s | --MB | -- img/s | --MB
 inception-v4| 77.64 | 94.93 | -- img/s | --MB | -- img/s | --MB
 - Training batch_size=16 for 2,0000 iterations, base_r=0.001 with 'poly' learning rate policy (power=0.9);
 - All the models use 513x513 input with random crop, multi-scale traing (0.75x, 1.0x, 1.25x, 1.5x, 2.0x) and horizantal flipping;
 - Testing with single scale, base_size=555 and crop_size=513, no flipping;
 - PSP module without batch normlization;
