# Classificaiton (caffe on ilsvrc)

This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem). But to achieve the original performance, finetuning is performed on ilsvrc dataset for several epoches. 

The main contribution belongs to the authors and model trainers.
