# Caffe-model
Python script to generate CNN models on Caffe, specially the inception\inception_v3\inception_v4\inception_resnet

# Models

The prototxts can be visualized by [ethereon](http://ethereon.github.io/netscope/quickstart.html).

Every model has a bn (batch normalization) version (maybe only bn version), the paper is [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3)

1. Lenet-5 (lenet.py)
 
   Lenet-5 was presented by Yann LeCun in [Backpropagation applied to handwritten zip code recognition](http://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf).


2. AlexNet (and caffenet in alexnet.py)
  
   AlexNet was initially described in [ImageNet Classification with Deep Convolutional
Neural Networks] (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

   Implemention of CaffeNet is referenced by [caffe/caffenet.py](https://github.com/BVLC/caffe/blob/master/examples/pycaffe/caffenet.py)
   
   
3. Network in network (nin.py)
 
   NIN model was described in [Network In Network](http://arxiv.org/pdf/1312.4400v3)


4. Inception_v1 (inception_v1.py)

   Inception conception was described in [Going Deeper with Convolutions](http://arxiv.org/pdf/1409.4842v1)
   
5. VggNet (vggnet.py)

   Vgg presented the network in [Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/pdf/1409.1556v6)
   
   The implemention of vgg_11a,vgg_11a_bn,vgg_16c,vgg_16c_bn are in vggnet.py
   
6. Inception_v3 (inception_v3.py)
 
   Inception_v3 is the improved version of inception_v1, the details are described in [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/pdf/1512.00567v3)

7. Inception_v4 (inception_resnet.py)

   Inception_v4 is is a more uniform simplified architecture and more inception modules than Inception-v3, the details are described in [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/pdf/1602.07261.pdf)

8. Inception_resnet (inception_resnet.py)
 
   Inception_resnet_v2 combines the residual connections and the latest revised version of the Inception architecture, single crop-single model top-5 error of inception_resnet_v2 is 4.9% on the non-blacklisted subset of the validation set of ILSVRC 2012. The details are described in [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/pdf/1602.07261.pdf)

9. ResNet

   Coming soon ......

# Acknowlegement

I greatly thank [Yangqing Jia](https://github.com/Yangqing) and [BVLC group](https://www.github.com/BVLC/caffe) for developing Caffe

And I would like to thank all the authors of every cnn model
