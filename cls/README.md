# Classificaiton (imagenet)

### Introduction
This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md), inception-resnet-v2 from [tensorflow](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)), some of them are converted from other modified caffe ([resnet-v2](https://github.com/yjxiong/caffe/tree/mem)). But to achieve the original performance, finetuning is performed on imagenet for several epoches. 

The main contribution belongs to the authors and model trainers.

### Performance on imagenet

Top-5 of pre-train models in this repository.
Network|Top-1|Top-5|Download|
:---:|:---:|:---:|:---:|:---:
resnet101-v2| 74.91| 92.19| [caffemodel (30.8  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCcHlfNmJkU2VPelE)| [netscope](http://ethereon.github.io/netscope/#/gist/4928834eca7f06261ba0558e0ff63a6a)
resnet152-v2| 76.09| 93.14| [caffemodel (54.6  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCRWVVdUJjVVAyQXc)| [netscope](http://ethereon.github.io/netscope/#/gist/71335b6e8634327c9b9216619572b3dd)
resnet269-v2| 77.31| 93.64| [caffemodel (77.3  MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCV3pud2oyR3lNMWs)| [netscope](http://ethereon.github.io/netscope/#/gist/ee808e19615844b8dbc7b13e92abd233)
inception-v3| 77.64| 93.79| [caffemodel (110 MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCa0phRGJIRERoTXM)| [netscope](http://ethereon.github.io/netscope/#/gist/8fae97d9c66b40b8da443f7f23e9b29b)
inception-resnet-v2| 77.64| 93.79| [caffemodel (110 MB)](https://drive.google.com/open?id=0B7ubpZO7HnlCa0phRGJIRERoTXM)| [netscope](http://ethereon.github.io/netscope/#/gist/8fae97d9c66b40b8da443f7f23e9b29b)

