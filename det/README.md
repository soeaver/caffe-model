
## Object Detection

**We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)
we are releasing the training code and files, the models and more experiments will come soon.**

### Object Detection Performance on PASCAL VOC. ([More experiments](https://github.com/soeaver/caffe-model/blob/master/det/VOC_Benchmark.md))

#### **1. Original Faster-RCNN training on VOC 2007+2012 trainval and testing on VOC 2007 test.**

 Network|mAP@50(%)|training<br/>speed|training<br/>memory|testing<br/>speed|testing<br/>memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 70.02 | 9.5 img/s | 1,235MB | 17.5 img/s | 989MB
 resnet101-v2| 79.6 | 3.1 img/s | 6,495MB | 7.1 img/s | 4,573MB
 wrn50-2| 78.59 | 2.1 img/s | 4,895MB | 4.9 img/s | 3,499MB
 resnext50-32x4d| 77.99 | 3.6 img/s | 5,315MB | 7.4 img/s | 4,305MB
 resnext101-32x4d| 79.98 | 2.7 img/s | 7,836MB | 6.3 img/s | 5,705MB
 inception-v4| 81.49 | 2.6 img/s | 6,759MB | 5.4 img/s | 4,683MB
 inception-resnet-v2| 80.0 | 2.0 img/s<br/> (batch=112) | 11,497MB | 3.2 img/s | 8,409MB
 air101| 81.0 | 2.4 img/s | 7,747MB | 5.1 img/s | 5,777MB
 
 - To reduce memory usage, we merge all the models batchnorm layer parameters into scale layer, more details please refer to [faster-rcnn-resnet](https://github.com/Eniac-Xie/faster-rcnn-resnet#modification) or [pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn/blob/master/tools/gen_merged_model.py);
 - We also split the deploy file to rpn deploy file and rcnn deploy file for adopting more testing tricks.
 - Performanc, speed and memory are calculated on [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv) with Nvidia Titan pascal, we do not guarantee that the results can be reproduced under any other conditions;
 - All the models are trained on a single scale (600*1000) with image flipping and train-batch=128 for 80,000 iterations, tested on the same single scale with test-batch=300 and nms=0.3;
 
 
#### **2. Faster-RCNN-2fc-OHEM training on VOC 2007+2012 trainval and testing on VOC 2007 test.**
 
  Network|mAP@50(%)|training<br/>speed|training<br/>memory|testing<br/>speed|testing<br/>memory
 :---:|:---:|:---:|:---:|:---:|:---:
 se-inception-v2 | (77.57) | 9.4 img/s | 2,453MB | 15.9 img/s | 1,573MB
 se-resnet50 | (79.73) | 6.2 img/s | 4,129MB | 12.8 img/s | 2,175MB
 resnet101-v2 | 80.6(80.49) | 5.0 img/s | 5,833MB | 10.5 img/s | 3,147MB
 air101 | (81.47) | 3.4 img/s | 6,653MB | 8.7 img/s | 4,503MB
 inception-v4-3x3 | 81.12(81.30) | 3.73 img/s | 5,383MB | 10.1 img/s | 3,217MB
 
 - 2fc means: conv256d --- fc1024d --- fc1024d;
 - The mAP@50 score in parentheses is training with ohem and [multigrid](https://arxiv.org/abs/1706.05587);
 
 
#### **3. RFCN-OHEM training on VOC 2007+2012 trainval and testing on VOC 2007 test.**

 Network|mAP@50(%)|training<br/>speed|training<br/>memory|testing<br/>speed|testing<br/>memory
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18 | 71.82 | 14.3 img/s | 1,215MB | 23.4 img/s | 899MB
 se-inception-v2| (78.23) | 10.2 img/s | 2,303MB | 14.0 img/s | 1,567MB
 se-resnet50 | (79.19) | 6.3 img/s | 3.999MB | 11.7 img/s | 2,205MB
 resnet101-v2| 78.93(79.9) | 4.9 img/s | 5,719MB | 10.4 img/s | 3,097MB
 resnext101-32x4d| 79.98(80.35) | 3.8 img/s | 6,977MB | 8.8 img/s | 4,761M
 air101| 79.42(80.93) | 3.4 img/s | 6,525MB | 8.5 img/s | 4,477MB
 inception-v4| 80.2 | 4.1 img/s | 4,371MB | 10.3 img/s | 2,343MB

 - The mAP@50 score in parentheses is training with ohem and [multigrid](https://arxiv.org/abs/1706.05587);
 
 
 ### Object Detection Performance on MSCOCO. ([More experiments](https://github.com/soeaver/caffe-model/blob/master/det/MSCOCO_Benchmark.md))
 
 #### **1. Results training on MSCOCO2017-trainval and testing on test-dev2017.**

 Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid | 32.6 | 53.6 | 34.5 | 12.5 | 35.1 | 48.4
 **RFCN-se-resnet50** <br/> with ms-train & ohem & multigrid | 32.9 | 54.4 | 34.8 | 13.0 | 35.3 | 48.1
 **RFCN-air101** <br/> with ms-train & ohem & multigrid | 38.2 | 60.1 | 41.2 | 18.2 | 41.9 | 53.0
 **Faster-2fc-air101** <br/> with ms-train & ohem & multigrid | 36.5 | 60.4 | 38.1 | 15.5 | 39.5 | 53.5
 
 - All the models are test on a single scale (600*1000) without any bells and whistles;
 
 
 #### **2. Context Pyramid Attention Network (CPANet) results training on MSCOCO2017-trainval and testing on test-dev2017.**
 
  Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test | 41.9 | 64.8 | 45.5 | 24.0 | 45.9 | 54.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms | 42.7 | 65.4 | 46.7 | 24.6 | 46.8 | 55.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms & flipping | 43.5 | 65.9 | 47.5 | 25.1 | 47.7 | 56.6
