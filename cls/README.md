# Classificaiton (imagenet)


### Introduction
This folder contains the deploy files(include generator scripts) and pre-train models of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2 and densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem)). But to achieve the original performance, finetuning is performed on imagenet for several epochs. 

The main contribution belongs to the authors and model trainers.

### Performance on imagenet
0. Top-1/5 accuracy of pre-train models in this repository.

 Network|224/299(single-crop)|224/299(12-crop)|320/395(single-crop)|320/395(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet101-v2| 78.05/93.88 | 80.01/94.96 | 79.63/94.84 | 80.71/95.43
 resnet152-v2| 79.15/94.58 | 80.76/95.32 | 80.34/95.26 | 81.16/95.68 
 resnet269-v2| **80.29**/95.00 | **81.75**/95.80 | 81.30/95.67 | **82.13**/96.15 
 inception-v3| 78.33/94.25 | 80.40/95.27 | 79.90/95.18 | 80.75/95.76 
 inception-v4| 79.97/94.91 | 81.40/95.70 | **81.32**/95.68 | 81.88/96.08 
 inception-resnet-v2| 80.14/**95.17** | 81.54/**95.92** | 81.25/**95.98** | 81.85/**96.29**
 resnext50_32x4d| 77.63/93.69 | 79.47/94.65 | 78.90/94.47 | 79.63/94.97 
 resnext101_32x4d| 78.70/94.21 | 80.53/95.11 | 80.09/95.03 | 80.81/95.41
 resnext101_64x4d| 79.40/94.59 | 81.12/95.41 | 80.74/95.37 | 81.52/95.69
 wrn50_2(resnet50_1x128d)| 77.87/93.87 | 79.91/94.94 | 79.32/94.72 | 80.17/95.13

 - The pre-train models are tested on original [caffe](https://github.com/BVLC/caffe) by [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py), **but ceil_mode:false（pooling_layer） is used for the models converted from torch, the detail in https://github.com/BVLC/caffe/pull/3057/files**. If you remove ceil_mode:false, the performance will decline about 1% top1.
 - 224x224(base_size=256) and 320x320(base_size=320) crop size for resnet-v2/resnext/wrn, 299x299(base_size=320) and 395x395(base_size=395) crop size for inception.

0. Top-1/5 accuracy with different crop sizes.
![teaser](https://github.com/soeaver/caffe-model/blob/master/cls/accuracy.png)
 - Figure: Accuracy curves of inception_v3(left) and resnet101_v2(right) with different crop sizes.

0. **Download url** and forward/backward time cost for each model.

 Forward/Backward time cost is evaluated with one image/mini-batch using cuDNN 5.1 on a Pascal Titan X GPU.
 
 We use
  ```
    ~/caffe/build/tools/caffe -model deploy.prototxt time -gpu -iterations 1000
  ```
 to test the forward/backward time cost, the result is really different with time cost of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py)

 Network|F/B(224/299)|F/B(320/395)|Download|Source
 :---:|:---:|:---:|:---:|:---:
 resnet101-v2| 22.31/22.75ms | 26.02/29.50ms | [170.3MB](https://pan.baidu.com/s/1kVQDHFx)|[craftGBD](https://github.com/craftGBD/craftGBD)
 resnet152-v2| 32.11/32.54ms | 37.46/41.84ms | [230.2MB](https://pan.baidu.com/s/1dFIc4vB)|[craftGBD](https://github.com/craftGBD/craftGBD)
 resnet269-v2| 58.20/59.15ms | 69.43/77.26ms | [390.4MB](https://pan.baidu.com/s/1qYbICs0)|[craftGBD](https://github.com/craftGBD/craftGBD)
 inception-v3| 21.79/19.82ms | 22.14/24.88ms | [91.1MB](https://pan.baidu.com/s/1boC0HEf)|[mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md)
 inception-v4| 32.96/32.19ms | 36.04/41.91ms | [163.1MB](https://pan.baidu.com/s/1c6D150)|[tensorflow_slim](https://github.com/tensorflow/models/tree/master/slim)
 inception-resnet-v2| 49.06/54.83ms | 54.06/66.38ms | [213.4MB](https://pan.baidu.com/s/1jHPJCX4)|[tensorflow_slim](https://github.com/tensorflow/models/tree/master/slim)
 resnext50_32x4d| 17.29/20.08ms | 19.02/23.81ms | [95.8MB](https://pan.baidu.com/s/1kVqgfJL)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101_32x4d| 30.73/35.75ms | 34.33/41.02ms | [169.1MB](https://pan.baidu.com/s/1hswrNUG)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101_64x4d| 42.07/64.58ms | 51.99/77.71ms | [319.2MB](https://pan.baidu.com/s/1pLhk0Zp)|[facebookresearch](https://github.com/facebookresearch/ResNeXt)
 wrn50_2(resnet50_1x128d)| 16.48/25.28ms | 20.99/35.04ms | [263.1MB](https://pan.baidu.com/s/1nvhoCsh)|[szagoruyko](https://github.com/szagoruyko/wide-residual-networks)

### Check the performance
0. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:
    ```
    ~/Database/ILSVRC2012
    ```

0. Check the resnet-v2 (101, 152 and 269) performance, the settings of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py):
   
    ```
    val_file = 'ILSVRC2012_val.txt' # download from this folder, label range 0~999
    ... ...
    model_weights = 'resnet-v2/resnet101_v2.caffemodel' # download as below
    model_deploy = 'resnet-v2/deploy_resnet101_v2.prototxt' # check the parameters of input_shape
    ... ...
    mean_value = np.array([102.9801, 115.9465, 122.7717])  # BGR
    std = np.array([1.0, 1.0, 1.0])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```

    Check the inception-v3 performance, the settings of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py):
   
    ```
    val_file = 'ILSVRC2015_val.txt' # download from this folder, label range 0~999
    ... ...
    model_weights = 'inception_v3/inception_v3.caffemodel' # download as below
    model_deploy = 'inception_v3/deploy_inception_v3.prototxt' # check the parameters of input_shape
    ... ...
    mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([128.0, 128.0, 128.0])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```
    
    Check the inception-resnet-v2 (inception-v4) performance, the settings of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py):
   
    ```
    val_file = 'ILSVRC2012_val.txt' # download from this folder, label range 0~999
    ... ...
    model_weights = 'inception_resnet_v2/inception_resnet_v2.caffemodel' # download as below
    model_deploy = 'inception_resnet_v2/deploy_inception_resnet_v2.prototxt' # check the parameters of input_shape
    ... ...
    mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([128.0, 128.0, 128.0])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```
    
    Check the resnext (50_32x4d, 101_32x4d and 101_64x4d) or wrn50_2 performance, the settings of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py):
   
    ```
    val_file = 'ILSVRC2012_val.txt' # download from this folder, label range 0~999
    ... ...
    model_weights = 'inception_resnet_v2/inception_resnet_v2.caffemodel' # download as below
    model_deploy = 'inception_resnet_v2/deploy_inception_resnet_v2.prototxt' # check the parameters of input_shape
    ... ...
    mean_value = np.array([103.52, 116.28, 123.675])  # BGR
    std = np.array([57.375, 57.12, 58.395])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```


0. then
    ```
    python evaluation_cls.py
    ```
