# Caffe-model
Caffe models (include classification, detection and segmentation) and deploy prototxt for resnet, resnext, inception_v3, inception_v4, inception_resnet, wider_resnet, densenet, aligned-inception-resne(x)t, DPNs and other networks.

Clone the caffe-model repository
  ```Shell
  git clone https://github.com/soeaver/caffe-model --recursive
  ```

## We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
Please install [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) for evaluating and finetuning.

## Disclaimer

Most of the pre-train models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[mxnet-model-gallery](https://github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、 [DenseNet](https://github.com/liuzhuang13/DenseNet)、 [wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)、 [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [ademxapp](https://github.com/itijyou/ademxapp)、 [DPNs](https://github.com/cypw/DPNs)、[Senet](https://github.com/hujie-frank/SENet)


## CLS (Classification, more details are in [cls](https://github.com/soeaver/caffe-model/tree/master/cls))
### Performance on imagenet validation.
**Top-1/5 error of pre-train models in this repository (Pre-train models download [urls](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation)).**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet101-v2| 21.95/6.12 | 19.99/5.04 | 20.37/5.16 | 19.29/4.57
 resnet152-v2| 20.85/5.42 | 19.24/4.68 | 19.66/4.73 | 18.84/4.32
 resnet269-v2| 19.71/5.00 | **18.25**/4.20 | 18.70/4.33 | **17.87**/3.85
 inception-v3| 21.67/5.75 | 19.60/4.73 | 20.10/4.82 | 19.25/4.24 
 xception| 20.90/5.49 | 19.68/4.90 | 19.58/4.77 | 18.91/4.39 
 inception-v4| 20.03/5.09 | 18.60/4.30 | 18.68/4.32 |18.12/3.92 
 inception-resnet-v2| 19.86/**4.83** | 18.46/**4.08** | 18.75/**4.02** | 18.15/**3.71**
 resnext50-32x4d| 22.37/6.31 | 20.53/5.35 | 21.10/5.53 | 20.37/5.03
 resnext101-32x4d| 21.30/5.79 | 19.47/4.89 | 19.91/4.97 | 19.19/4.59
 resnext101-64x4d| 20.60/5.41 | 18.88/4.59 | 19.26/4.63 | 18.48/4.31
 wrn50-2<br/>(resnet50-1x128d)| 22.13/6.13 | 20.09/5.06 | 20.68/5.28 | 19.83/4.87
 air101| 21.32/5.76 | 19.36/4.84 | 19.92/4.75 | 19.05/4.43
 dpn-92| 20.81/5.47 | 18.99/4.59 | 19.23/4.64 | 18.68/4.24
 dpn-107| 19.70/5.06 | ../.. | 18.41/4.25 | ../..
 

## DET (Detection, more details are in [det](https://github.com/soeaver/caffe-model/tree/master/det))
### Object Detection Performance on PASCAL VOC.
**Original faster rcnn train on VOC 2007+2012 trainval and test on VOC 2007 test.**

 Network|mAP@50|train speed|train memory|test speed|test memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 70.02 | 9.5 img/s | 1,235MB | 17.5 img/s | 989MB
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | 7.1 img/s | 4,573MB
 resnet152-v2| 80.72 | 2.8 img/s | 9,315MB | 6.2 img/s | 6,021MB
 wrn50-2| 78.59 | 2.1 img/s | 4,895MB | 4.9 img/s | 3,499MB
 resnext50-32x4d| 77.99 | 3.6 img/s | 5,315MB | 7.4 img/s | 4,305MB
 resnext101-32x4d| 79.98 | 2.7 img/s | 7,836MB | 6.3 img/s | 5,705MB
 resnext101-64x4d| 80.71 | 2.0 img/s<br/> (batch=96) | 11,277MB | 3.7 img/s | 9,461MB
 inception-v3| 78.6 | 4.1 img/s | 4,325MB | 7.3 img/s | 3,445MB
 inception-v4| 81.49 | 2.6 img/s | 6,759MB | 5.4 img/s | 4,683MB
 inception-resnet-v2| 80.0 | 2.0 img/s<br/> (batch=112) | 11,497MB | 3.2 img/s | 8,409MB
 densenet-201| 77.53 | 3.9 img/s<br/> (batch=72) | 10,073MB | 5.5 img/s | 9,955MB
 resnet38a| 80.1 | 1.4 img/s | 8,723MB | 3.4 img/s | 5,501MB
 
 
## SEG (Segmentation, more details are in [seg](https://github.com/soeaver/caffe-model/tree/master/seg))
### Object Segmentation Performance on PASCAL VOC.
**PSPNet training on [SBD](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf) (10,582 images) and testing on VOC 2012 validation (1,449 images).**

 Network|mIoU(%)|pixel acc(%)|training<br/>speed|training<br/>memory|testing<br/>speed|testing<br/>memory
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 resnet101-v2| 77.94 | 94.94 | 1.6 img/s | 8,023MB | 3.0 img/s | 4,071MB
 resnet101-v2-selu| 77.10 | 94.80 | 1.6 img/s | 8,017MB | 3.0 img/s | 4,065MB
 resnext101-32x4d| 77.79 | 94.92 | 1.3 img/s | 8,891MB | 2.6 img/s | 5,241MB
 air101| 77.64 | 94.93 | 1.3 img/s | 10,017MB | 2.5 img/s | 5,241MB
 inception-v4| 77.58 | 94.83 | -- img/s | --MB | -- img/s | --MB
 

## License

caffe-model is released under the MIT License (refer to the LICENSE file for details).


## Acknowlegement

I greatly thank [Yangqing Jia](https://github.com/Yangqing) and [BVLC group](https://www.github.com/BVLC/caffe) for developing Caffe.

And I would like to thank all the authors of every network.
