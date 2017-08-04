# Caffe-model
Caffe models (include classification, detection and segmentation) and deploy prototxt for resnet, resnext, inception_v3, inception_v4, inception_resnet, wider_resnet, densenet, aligned-inception-resne(x)t, DPNs and other networks.

## We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
Please install [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) for evaluating and finetuning.

## Disclaimer

Most of the pre-train models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[mxnet-model-gallery](https://github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、 [DenseNet](https://github.com/liuzhuang13/DenseNet)、 [wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)、 [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [ademxapp](https://github.com/itijyou/ademxapp)、 [DPNs](https://github.com/cypw/DPNs)


## CLS (Classification, more details are in [cls](https://github.com/soeaver/caffe-model/tree/master/cls))
### Performance on imagenet validation.
**Top-1/5 error of pre-train models in this repository (Pre-train models download [urls](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation)).**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv| 29.11/10.07 | 26.69/8.64 | 27.54/8.98 | 26.23/8.21
 resnext26-32x4d-priv| 24.93/7.75 | 23.54/6.89 | 24.20/7.21 | 23.19/6.60
 resnet101-v2| 21.95/6.12 | 19.99/5.04 | 20.37/5.16 | 19.29/4.57
 resnet152-v2| 20.85/5.42 | 19.24/4.68 | 19.66/4.73 | 18.84/4.32
 resnet269-v2| 19.71/5.00 | **18.25**/4.20 | 18.70/4.33 | **17.87**/3.85
 resnet38a| 20.66/5.27 | ../.. | 19.25/4.66 | ../..
 inception-v3| 21.67/5.75 | 19.60/4.73 | 20.10/4.82 | 19.25/4.24 
 xception| 20.90/5.49 | 19.68/4.90 | 19.58/4.77 | 18.91/4.39 
 inception-v4| 20.03/5.09 | 18.60/4.30 | 18.68/4.32 |18.12/3.92 
 inception-resnet-v2| 19.86/**4.83** | 18.46/**4.08** | 18.75/**4.02** | 18.15/**3.71**
 resnext50-32x4d| 22.37/6.31 | 20.53/5.35 | 21.10/5.53 | 20.37/5.03
 resnext101-32x4d| 21.30/5.79 | 19.47/4.89 | 19.91/4.97 | 19.19/4.59
 resnext101-64x4d| 20.60/5.41 | 18.88/4.59 | 19.26/4.63 | 18.48/4.31
 wrn50-2<br/>(resnet50-1x128d)| 22.13/6.13 | 20.09/5.06 | 20.68/5.28 | 19.83/4.87
 air101<br/>(aligned-inception-resnet101)| 20.74/5.56 | ../.. | ../.. | ../..
 dpn-92| 20.81/5.47 | 18.99/4.59 | 19.23/4.64 | ../.. 
 dpn-98| 20.27/5.28 | ../.. | 18.87/4.43 | ../..
 dpn-131| 20.00/5.24 | ../.. | 18.63/4.31 | ../..
 dpn-107| **19.70**/5.06 | ../.. | **18.41**/4.25 | ../..
 
 - The resnet18-priv, resnext26-32x4d-priv are trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.
 - The pre-train models are tested on original [caffe](https://github.com/BVLC/caffe) by [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py), **but ceil_mode:false（pooling_layer） is used for the models converted from torch, the detail in https://github.com/BVLC/caffe/pull/3057/files**. If you remove ceil_mode:false, the performance will decline about 1% top1.
 - 224x224(base_size=256) and 320x320(base_size=320) crop size for resnet-v2/resnext/wrn, 299x299(base_size=320) and 395x395(base_size=395) crop size for inception.

## DET (Detection, more details are in [det](https://github.com/soeaver/caffe-model/tree/master/det))
### Object Detection Performance on PASCAL VOC.
**Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|mAP@50|train speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 70.02 | 9.5 img/s | 1,235MB | 17.5 img/s | 989MB
 resnet101| -- | -- | -- | -- | --
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | 7.1 img/s | 4,573MB
 resnet152-v2| 80.72 | 2.8 img/s | 9,315MB | 6.2 img/s | 6,021MB
 wrn50-2| 78.59 | 2.1 img/s | 4,895MB | 4.9 img/s | 3,499MB
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
 

## License

caffe-model is released under the MIT License (refer to the LICENSE file for details).


## Acknowlegement

I greatly thank [Yangqing Jia](https://github.com/Yangqing) and [BVLC group](https://www.github.com/BVLC/caffe) for developing Caffe.

And I would like to thank all the authors of every network.
