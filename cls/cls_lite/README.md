## CLS Lite (Classification lite)

Please install [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) for evaluating and finetuning.


### Performance of lite models on imagenet validation.
**1. Top-1/5 error and CPU/GPU speed of lite models in this repository.**

 Network|Top-1/5 error|F/B on GPU|F/B on CPU|Source
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv | 29.11/10.07 | 4.48/5.07ms | 213.2/193.3ms | [pytorch-classification](https://github.com/soeaver/pytorch-classification)
 resnet18-1x96d | 26.11/8.31 | 6.16/9.94ms | 443.2/419.0ms | [pytorch-classification](https://github.com/soeaver/pytorch-classification)
 resnet18-1x128d | 24.81/7.61 | 9.75/16.94ms | 729.1/695.4ms | [pytorch-classification](https://github.com/soeaver/pytorch-classification)
 vgg13-pytorch | 31.07/11.13 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg13bn-pytorch | 29.50/10.18 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16-pytorch | 29.14/10.00 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16-tf | -- | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16-dsd | -- | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16bn-pytorch | 27.53/8.99 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg19-pytorch | 28.23/9.60 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg19bn-pytorch | 26.58/8.45 | 9.75/16.94ms | 729.1/695.4ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
