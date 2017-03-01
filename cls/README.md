# Classificaiton (imagenet)

### Introduction
This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem)). But to achieve the original performance, finetuning is performed on imagenet for several epoches. 

The main contribution belongs to the authors and model trainers.

### Performance on imagenet

Top-5 of pre-train models in this repository.

Network|224/299(single-crop)|224/299(12-crop)|320/331(single-crop)|320/331(12-crop)
:---:|:---:|:---:|:---:|:---:
resnet101-v2| 74.91| 92.19| [caffemodel (30.8  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCcHlfNmJkU2VPelE)| 0
resnet152-v2| 76.09| 93.14| [caffemodel (54.6  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCRWVVdUJjVVAyQXc)| 0
resnet269-v2| 77.31| 93.64| [caffemodel (77.3  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCV3pud2oyR3lNMWs)| 0
inception-v3| 77.64| 93.79| [caffemodel (110 MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCa0phRGJIRERoTXM)| 0
inception-resnet-v2| 77.64| 93.79| [caffemodel (110 MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCa0phRGJIRERoTXM)| 0

