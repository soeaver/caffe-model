## CLS Lite (Classification lite)

Please install [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) for evaluating and finetuning.


### Performance of lite models on imagenet validation.
**1. Top-1/5 error and CPU/GPU speed of lite models in this repository.**

 Network|Top-1/5 error|F/B on GPU|F/B on CPU|Source
 :---:|:---:|:---:|:---:|:---:
 resnet10-1x32d | 44.78/21.42 | 2.19/2.57ms | 42.84/38.00ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet10-1x48d | -- | 2.55/3.01ms | 83.66/75.97ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet10-1x64d | 35.93/14.59 | 2.93/3.86ms | 134.3/124.8ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet10-1x96d | 30.66/11.13 | 3.42/5.57ms | 220.7/204.9ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x16d | 51.37/26.35 | 3.03/3.22ms | 25.03/22.63ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x32d | 38.24/16.02 | 3.53/4.14ms | 69.2/63.2ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x48d | 32.55/11.87 | 4.30/4.83ms | 139.1/127.6ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x64d<br/>(resnet18-priv) | 29.62/10.38 | 4.48/5.07ms | 213.2/193.3ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x96d | 26.11/8.31 | 6.16/9.94ms | 443.2/419.0ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnet18-1x128d | 24.81/7.61 | 9.75/16.94ms | 729.1/695.4ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnext26-32x4d | 25.57/8.12 | 9.68/11.16ms | 331.4/300.2ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 vgg13-pytorch | 31.07/11.13 | 5.70/9.35ms | 1318/1279ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg13bn-pytorch | 29.50/10.18 | 8.35/13.49ms | 1443/1336ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16-pytorch | 29.14/10.00 | 6.79/11.78ms | 1684/1643ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg16-tf | 29.03/10.12 | 13.04/48.90ms | 1787/1647ms | [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)
 vgg16-dsd | 27.62/9.02 | 6.81/11.80ms | 1753/1660ms | [dsd](https://github.com/songhan/DSD)
 vgg16-5x | 31.67/11.60 | 4.46/7.15ms | 580.5/593.0ms | [channel-pruning](https://github.com/yihui-he/channel-pruning)
 vgg16-3c4x | 28.79/9.78 | 7.53/9.77ms | 753.4/772.4ms | [channel-pruning](https://github.com/yihui-he/channel-pruning)
 vgg16bn-pytorch | 27.53/8.99 | 9.14/15.83ms | 1783/1695ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg19-pytorch | 28.23/9.60 | 8.03/14.26ms | 2076/2012ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 vgg19bn-pytorch | 26.58/8.45 | 10.75/18.77ms | 2224/2081ms | [vision](https://github.com/pytorch/vision/tree/master/torchvision/models)
 inception-v1-tf | 31.37/11.10 | 10.66/7.84ms | 186.2/155.8ms | [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)
 inception-v2-tf | 27.91/9.40 | 13.93/10.65ms | 286.4/255.0ms | [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)
 xception | 20.90/5.49 | 15.21/31.65ms | 1262/1253ms | [keras-models](https://github.com/fchollet/deep-learning-models)
 mobilenet-v1-1.0 | 29.98/10.52 | 6.16/9.50ms | 169.4/138.1ms | [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)
 air14-1x8d | 56.28/31.25 | 4.28/3.08ms | 21.01/3.29ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air14-1x16d | 44.23/20.68 | 5.13/3.56ms | 45.45/6.41ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air26-1x16d | 36.31/14.59 | 7.32/4.70ms | 62.02/8.52ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air26-1x32d | 28.71/9.59 | 8.77/5.05ms | 170.7/19.25ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air50-1x16d | 31.19/11.26 | 14.73/8.31ms | 91.65/16.06ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 air50-1x32d | 25.59/7.89 | 15.39/7.64ms | 229.6/22.81ms | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 dpn68 | 22.56/6.24 | 22.70/21.41ms | 371.1/329.3ms | [DPNs](https://github.com/cypw/DPNs) 
 se-resnet50 | 22.39/6.37 | 17.91/19.49ms | 932.2/821.4ms | [senet](https://github.com/hujie-frank/SENet) 
 se-resnet50-hik | 21.98/5.80 | 17.43/20.13ms | 581.1/482.7ms | [senet-caffe](https://github.com/shicai/SENet-Caffe) 
 se-inception-v2 | 23.64/7.04 | 15.31/11.21ms | 251.9/218.5ms | [senet](https://github.com/hujie-frank/SENet) 
