# Classificaiton (imagenet)

### Introduction
This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem)). But to achieve the original performance, finetuning is performed on imagenet for several epoches. 

The main contribution belongs to the authors and model trainers.

### Performance on imagenet
Top-1/Top-5 of pre-train models in this repository.

Network|224/299(single-crop)|224/299(12-crop)|320/331(single-crop)|320/331(12-crop)
:---:|:---:|:---:|:---:|:---:
resnet101-v2| 77.4/93.5 | 79.2/94.6 | 79.6/94.9 | 80.4/95.4 
resnet152-v2| -- | -- | -- | -- 
resnet269-v2| 79.1/94.4 | 80.5/95.2 | **81.1/95.7** | -- 
inception-v3| 77.6/93.9 | 79.0/94.6 | 78.6/94.4 | 79.9/95.1 
inception-resnet-v2| 79.6/94.9 | 80.7/95.6 | 80.5/95.5 | -- 

- All the pre-train models are tested on origial [caffe](https://github.com/BVLC/caffe) by [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py).
- 224x224(short_size=256) and 320x320 crop size for resnet-v2, 299x299(short_size=320) and 331x331 crop size for inception.
- The models are uploading, please wait.
- -- means that I have not done this test yet.

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
    class_num = 1000
    ... ...
    mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([1.0, 1.0, 1.0])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```

0. Check the inception-v3 performance, the settings of [evaluation_cls.py](https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py):
   
    ```
    val_file = 'ILSVRC2015_val.txt' # download from this folder, label range 1~1000
    ... ...
    model_weights = 'inception_v3/inception_v3.caffemodel' # download as below
    model_deploy = 'inception_v3/deploy_inception_v3.prototxt' # check the parameters of input_shape
    ... ...
    class_num = 1008
    ... ...
    mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([1.0, 1.0, 1.0])  # BGR
    crop_num = 1    # perform center(single)-crop
    ```
    
0. then
    ```
    python evaluation_cls.py
    ```
