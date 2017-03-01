# Classificaiton (caffe on ilsvrc)

This folder contains the deploy files(include generator scripts) and caffe models(coming soon) of resnet-v1, resnet-v2, inception-v3, inception-resnet-v2, densenet(coming soon).

We didn't train any model from scratch, some of them are converted from other deep learning framworks (inception-v3 from mxnet, inception-resnet-v2 from tensorflow), some of them are converted from other modified caffe (resnet-v2). But to achieve the original performance, finetuning is performed on ilsvrc dataset for several epoches. 
