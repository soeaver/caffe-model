# Classificaiton (imagenet)

### Introduction
This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem)). But to achieve the original performance, finetuning is performed on imagenet for several epoches. 

The main contribution belongs to the authors and model trainers.

### Performance on imagenet

Top-5 of pre-train models in this repository.

Network|224/299(single-crop)|224/299(12-crop)|320/331(single-crop)|320/331(12-crop)
:---:|:---:|:---:|:---:|:---:
resnet101-v2| 93.5 | 94.5 | 94.9 | 95.4 
resnet152-v2| -- | -- | -- | -- 
resnet269-v2| 94.4 | 95.2 | 95.6 | -- 
inception-v3| 93.8 | 94.6 | 94.3 | 95.0 
inception-resnet-v2| 94.9 | -- | 95.4 | -- 

- 224x224 and 320x320 crop size for resnet-v2, 299x299 and 331x331 crop size for inception.

