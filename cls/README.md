## CLS (Classification)

Please install [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) for evaluating and finetuning.

### Disclaimer

Most of the models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[mxnet-model-gallery](https://github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、 [DenseNet](https://github.com/liuzhuang13/DenseNet)、 [wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)、 [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [ademxapp](https://github.com/itijyou/ademxapp)、 [DPNs](https://github.com/cypw/DPNs)、[Senet](https://github.com/hujie-frank/SENet)


### Performance on imagenet validation.
**1. Top-1/5 error of pre-train models in this repository.**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv| 29.62/10.38 | 26.69/8.64 | 27.54/8.98 | 26.23/8.21
 resnext26-32x4d-priv| 24.93/7.75 | 23.54/6.89 | 24.20/7.21 | 23.19/6.60
 resnet101-v2| 21.95/6.12 | 19.99/5.04 | 20.37/5.16 | 19.29/4.57
 resnet152-v2| 20.85/5.42 | 19.24/4.68 | 19.66/4.73 | 18.84/4.32
 resnet269-v2| 19.71/5.00 | 18.25/4.20 | 18.70/4.33 | 17.87/3.85
 resnet38a| 20.66/5.27 | ../.. | 19.25/4.66 | ../..
 inception-v3| 21.67/5.75 | 19.60/4.73 | 20.10/4.82 | 19.25/4.24 
 xception| 20.90/5.49 | 19.68/4.90 | 19.58/4.77 | 18.91/4.39 
 inception-v4| 20.03/5.09 | 18.60/4.30 | 18.68/4.32 |18.12/3.92 
 inception-resnet-v2| 19.86/4.83 | 18.46/4.08 | 18.75/4.02 | 18.15/3.71
 resnext50-32x4d| 22.37/6.31 | 20.53/5.35 | 21.10/5.53 | 20.37/5.03
 resnext101-32x4d| 21.30/5.79 | 19.47/4.89 | 19.91/4.97 | 19.19/4.59
 resnext101-64x4d| 20.60/5.41 | 18.88/4.59 | 19.26/4.63 | 18.48/4.31
 wrn50-2<br/>(resnet50-1x128d)| 22.13/6.13 | 20.09/5.06 | 20.68/5.28 | 19.83/4.87
 airx50-24x4d| 22.39/6.23 | 20.36/5.19 | 20.88/5.33 | 19.97/4.92
 air101| 21.32/5.76 | 19.36/4.84 | 19.92/4.75 | 19.05/4.43
 air152| 20.38/5.11 | 18.46/4.26 | 19.08/4.40 | 18.53/4.00
 airx101-32x4d| 21.15/5.74 | 19.43/4.86 | 19.61/4.93 | 18.90/4.49
 dpn-68-extra| 22.56/6.24 | 20.48/4.99 | 20.99/5.25 | 20.09/4.73
 dpn-92| 20.81/5.47 | 18.99/4.59 | 19.23/4.64 | 18.68/4.24 
 dpn-98| 20.27/5.28 | 18.57/4.42 | 18.87/4.43 | 18.21/4.11
 dpn-131| 20.00/5.24 | 18.52/4.28 | 18.63/4.31 | 17.99/3.92
 dpn-107-extra| 19.70/5.06 | ../.. | 18.41/4.25 | ../..
 se-inception-v2<br/>(se-inception-bn)| 23.64/7.04 | 21.57/5.86 | 21.61/5.87 | 20.85/5.38
 se-resnet50| 22.39/6.37 | 20.61/5.34 | 20.49/5.22 | 20.02/4.85
 se-resnet50-hik| 21.98/5.80 | 20.06/4.88 | 20.51/5.04 | 19.92/4.68
 se-resnet101| 21.76/5.72 | 19.96/4.79 | 19.97/4.78 | 19.34/4.41
 se-resnet152| 21.34/5.54 | 19.56/4.66 | 19.34/4.59 | 18.83/4.32
 se-resnext50-32x4d| 20.96/5.53 | 19.39/4.69 | 19.36/4.66 | 18.70/4.38
 se-resnext101-32x4d| 19.83/4.95 | 18.44/4.16 | 18.14/4.08 | 17.68/3.86
 senet<br/>(se-resnext152-64x4d)| 18.67/4.47 | 17.40/3.69 | 17.28/3.78 | 16.80/3.47

 - The resnet18-priv, resnext26-32x4d-priv are trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.
 - The pre-train models are tested on original [caffe](https://github.com/BVLC/caffe) by [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py), **but ceil_mode:false（pooling_layer） is used for the models converted from torch, the detail in https://github.com/BVLC/caffe/pull/3057/files**. If you remove ceil_mode:false, the performance will decline about 1% top1.
 - 224x224(base_size=256) and 320x320(base_size=320) crop size for resnet-v2/resnext/wrn, 299x299(base_size=320) and 395x395(base_size=395) crop size for inception.

**2. Top-1/5 accuracy with different crop sizes.**
![teaser](https://github.com/soeaver/caffe-model/blob/master/cls/accuracy.png)
 - Figure: Accuracy curves of inception_v3(left) and resnet101_v2(right) with different crop sizes.

**3. Download url and forward/backward time cost for each model.**

 Forward/Backward time cost is evaluated with one image/mini-batch using cuDNN 5.1 on a Pascal Titan X GPU.
 
 We use
  ```
    ~/caffe/build/tools/caffe -model deploy.prototxt time -gpu -iterations 1000
  ```
 to test the forward/backward time cost, the result is really different with time cost of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py)

 Network|F/B(224/299)|F/B(320/395)|Download<br/>(BaiduCloud)|Download<br/>(GoogleDrive)|Source
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18-priv | 4.48/5.07ms | 4.99/7.01ms | [44.6MB](https://pan.baidu.com/s/1hrYc3La)|44.6MB|[pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnext26-32x4d-priv | 8.53/10.12ms | 10.55/13.46ms | [58.9MB](https://pan.baidu.com/s/1dFzmUOh)|[58.9MB](https://drive.google.com/open?id=0B9mkjlmP0d7zZEh4dzZ3TVZUb2M)|[pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet101-v2| 22.31/22.75ms | 26.02/29.50ms | [170.3MB](https://pan.baidu.com/s/1kVQDHFx)|[170.3MB](https://drive.google.com/open?id=0B9mkjlmP0d7zRlhISks0VktGOGs)|[craftGBD](https://github.com/craftGBD/craftGBD)
 resnet152-v2| 32.11/32.54ms | 37.46/41.84ms | [230.2MB](https://pan.baidu.com/s/1dFIc4vB)|[230.2MB](https://drive.google.com/open?id=0B9mkjlmP0d7zOXhrb1EyYVRHOEk)|[craftGBD](https://github.com/craftGBD/craftGBD)
 resnet269-v2| 58.20/59.15ms | 69.43/77.26ms | [390.4MB](https://pan.baidu.com/s/1qYbICs0)|[390.4MB](https://drive.google.com/open?id=0B9mkjlmP0d7zOGFxcTMySHN6bUE)|[craftGBD](https://github.com/craftGBD/craftGBD)
 inception-v3| 21.79/19.82ms | 22.14/24.88ms | [91.1MB](https://pan.baidu.com/s/1boC0HEf)|[91.1MB](https://drive.google.com/open?id=0B9mkjlmP0d7zTEJmNEh6c0RfYzg)|[mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md)
 xception | 14.03/30.39ms | 19.46/48.64ms | [87.4MB](https://pan.baidu.com/s/1gfiTShd)|87.4MB|[keras-models](https://github.com/fchollet/deep-learning-models)
 inception-v4| 32.96/32.19ms | 36.04/41.91ms | [163.1MB](https://pan.baidu.com/s/1c6D150)|[163.1MB](https://drive.google.com/open?id=0B9mkjlmP0d7zUEJ3aEJ2b3J0RFU)|[tf-slim](https://github.com/tensorflow/models/tree/master/slim)
 inception-resnet-v2| 49.06/54.83ms | 54.06/66.38ms | [213.4MB](https://pan.baidu.com/s/1jHPJCX4)|[213.4MB](https://drive.google.com/open?id=0B9mkjlmP0d7zc3A4NWlQQzdoM28)|[tf-slim](https://github.com/tensorflow/models/tree/master/slim)
 resnext50-32x4d| 17.29/20.08ms | 19.02/23.81ms | [95.8MB](https://pan.baidu.com/s/1kVqgfJL)|[95.8MB](https://drive.google.com/open?id=0B9mkjlmP0d7zYVgwanhVWnhrYlE)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101-32x4d| 30.73/35.75ms | 34.33/41.02ms | [169.1MB](https://pan.baidu.com/s/1hswrNUG)|[169.1MB](https://drive.google.com/open?id=0B9mkjlmP0d7zTzYyelgyYlpOU3c)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101-64x4d| 42.07/64.58ms | 51.99/77.71ms | [319.2MB](https://pan.baidu.com/s/1pLhk0Zp)|[319.2MB](https://drive.google.com/open?id=0B9mkjlmP0d7zQ0ZZOENnSFdQWnc)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 wrn50-2<br/>(resnet50_1x128d)| 16.48/25.28ms | 20.99/35.04ms | [263.1MB](https://pan.baidu.com/s/1nvhoCsh)|[263.1MB](https://drive.google.com/open?id=0B9mkjlmP0d7zYW40dUMxS3VPclU)|[szagoruyko](https://github.com/szagoruyko/wide-residual-networks)
 airx50-24x4d| 23.59/24.80ms | 26.64/30.92ms | .. | .. |[pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air101| 35.78/35.94ms | 39.69/45.52ms | .. | .. |[pytorch-cls](https://github.com/soeaver/pytorch-classification)
 airx101-32x4d| 49.43/55.52ms | 54.64/66.31ms | .. | .. |[pytorch-cls](https://github.com/soeaver/pytorch-classification)
 dpn-68| ../..ms | ../..ms | [48.4MB](https://pan.baidu.com/s/1bphINV5) | .. |[DPNs](https://github.com/cypw/DPNs)
 dpn-92| 29.71/30.68ms | 35.19/37.13ms  | [144.2MB](https://pan.baidu.com/s/1pL0VuWV)|[144.2MB](https://drive.google.com/open?id=0B9mkjlmP0d7zaWVKWFd2OXpRTVU)|[DPNs](https://github.com/cypw/DPNs)
 dpn-98| 36.24/44.06ms | 42.84/53.50ms | [235.6MB](https://pan.baidu.com/s/1pKHBRlD) | .. |[DPNs](https://github.com/cypw/DPNs)
 dpn-107| 45.21/59.77ms | 56.12/77.78ms | [332.4MB](https://pan.baidu.com/s/1i5b0Uih) | .. |[DPNs](https://github.com/cypw/DPNs)
 dpn-131| 48.20/59.43ms | 57.66/72.43ms | [303.3MB](https://pan.baidu.com/s/1miOdMHi) | .. |[DPNs](https://github.com/cypw/DPNs)
 se-inception-v2| 14.66/10.63ms | 15.71/13.52ms | .. | .. |[senet](https://github.com/hujie-frank/SENet)
 se-resnet50| 15.29/14.20ms | 17.96/19.69ms | .. | .. |[senet](https://github.com/hujie-frank/SENet)
  
 - For speeding up xception, we adopt [convolution depthwise layer](https://github.com/BVLC/caffe/pull/5665/files).

### Check the performance
**1. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:**

      ~/Database/ILSVRC2012

**2. Modify the parameter settings**

 Network|val_file|mean_value|std
 :---:|:---:|:---:|:---:
 resnet-v2(101/152/269)| ILSVRC2012_val | [102.98, 115.947, 122.772] | [1.0, 1.0, 1.0]
 resnet10/18/, resnext, air(x) | ILSVRC2012_val | [103.52, 116.28, 123.675] | [57.375, 57.12, 58.395]
 inception-v3| **ILSVRC2015_val** | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0] 
 inception-v2, xception<br/>inception-v4, inception-resnet-v2 | ILSVRC2012_val | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0] 
 dpn(68/92/98/131/107)| ILSVRC2012_val | [104.0, 117.0, 124.0] | [59.88, 59.88, 59.88]
 official senet| **ILSVRC2015_val** | [104.0, 117.0, 123.0] | [1.0, 1.0, 1.0] 


**3. then run evaluation_cls.py**

    python evaluation_cls.py
