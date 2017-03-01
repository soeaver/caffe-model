### Resnet-v2

At present, We have not finished the generator scripts of resnet-v2 yet. Maybe [ResNet_with_IdentityMapping](https://github.com/MichaelHunson/ResNet_with_IdentityMapping) is useful.

The detail is described in the paper **Identity Mappings in Deep Residual Networks** (https://arxiv.org/abs/1603.05027).

The caffe models are converted from **craftGBD** (https://github.com/craftGBD/craftGBD). 
These models in craftGBD are different in BN layer, We manually converted the modified 'bn_layer' to offical 'batch_norm_layer and scale_layer'.

