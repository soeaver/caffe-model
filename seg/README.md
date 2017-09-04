## Object Segmentation

### We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
we are releasing the training code and files, the models and more experiments will come soon.

### Object Segmentation Performance on PASCAL VOC.
**1. PSPNet training on [SBD](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf) (10,582 images) and testing on VOC 2012 validation (1,449 images).**

 Network|mIoU(%)|pixel acc(%)|training<br/>speed|training<br/>memory|testing<br/>speed|testing<br/>memory
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 resnet101-v2| 77.94 | 94.94 | 1.6 img/s | 8,023MB | 3.0 img/s | 4,071MB
 resnet101-v2-selu| 77.10 | 94.80 | 1.6 img/s | 8,017MB | 3.0 img/s | 4,065MB
 resnext101-32x4d| 77.79 | 94.92 | 1.3 img/s | 8,891MB | 2.6 img/s | 5,241MB
 air101| 77.64 | 94.93 | 1.3 img/s | 10,017MB | 2.5 img/s | 5,241MB
 inception-v4| 77.58 | 94.83 | -- img/s | --MB | -- img/s | --MB
 se-resnet50| 75.80 | 94.30 | -- img/s | --MB | -- img/s | --MB
 - To reduce memory usage, we merge all the models batchnorm layer parameters into scale layer, more details please refer to [faster-rcnn-resnet](https://github.com/Eniac-Xie/faster-rcnn-resnet#modification) or [pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn/blob/master/tools/gen_merged_model.py);
 - PSP module without batch normlization, the kernel_size of avepooling is 64, 32, 16 and 8 respectively;
 - All the models use 513x513 input with random crop, multi-scale traing (0.75x, 1.0x, 1.25x, 1.5x, 2.0x) and horizantal flipping;
 - The training and testing speed is calculated on a single Nvidia Titan pascal GPU with batch_size=1;
 - Training batch_size=16 for 2,0000 iterations, base_lr=0.001 with 'poly' learning rate policy (power=0.9);
 - Testing with single scale, base_size=555 and crop_size=513, no flipping, no crf;
