## MSCOCO Detection Benchmark

**We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)**

### **1. Results training on MSCOCO2017-trainval and testing on test-dev2017.**

 Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid | 32.6 | 53.6 | 34.5 | 12.5 | 35.1 | 48.4
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping & ms-test | 36.8 | 59.8 | 38.7 | 19.7 | 39.8 | 49.1
 **RFCN-se-resnet50** <br/> with ms-train & ohem & multigrid | 32.9 | 54.4 | 34.8 | 13.0 | 35.3 | 48.1
 **FPN-Faster-inception-v4** <br/> with ms-train | 36.5 | 58.5 | 38.8 | 16.5 | 38.8 | 52.1
 **FPN-Faster-inception-v4** <br/> with ms-train & bbox-voting & soft-nms | 38.3 | 61.0 | 40.8 | 20.0 | 41.5 | 51.4
 **FPN-Faster-inception-v4** <br/> with ms-train & bbox-voting & soft-nms & flipping & ms-test | 39.5 | 62.5 | 42.3 | 23.3 | 43.2 | 51.0
 **RFCN-air101** <br/> with ms-train & ohem & multigrid | 38.2 | 60.1 | 41.2 | 18.2 | 41.9 | 53.0
 **RFCN-air101** <br/> with extra-7-epochs & ms-train & ohem & multigrid  | 38.5 | 60.2 | 41.4 | 18.3 | 42.1 | 53.4
 **RFCN-air101** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping | 40.4 | 63.5 | 43.5 | 22.6 | 44.4 | 52.0
 **RFCN-air101** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping & ms-test | 41.8 | 65.3 | 45.3 | 26.1 | 45.6 | 52.4
 **RFCN-air101** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping & assign-ms-test | 42.1 | 64.6 | 45.6 | 25.6 | 44.5 | 54.1
 **RFCN-air101** <br/> with ms-train & ohem & multigrid & deformpsroi & bbox-voting & soft-nms & flipping & assign-ms-test | 43.2 | 66.0 | 46.7 | 25.6 | 46.3 | 55.9
 **Faster-2fc-air101** <br/> with ms-train & ohem & multigrid | 36.5 | 60.4 | 38.1 | 15.5 | 39.5 | 53.5
 
 - All the models are test on a single scale (600*1000) without any bells and whistles;
  
 
### **2. Context Pyramid Attention Network (CPANet) results training on MSCOCO2017-trainval and testing on test-dev2017.**
 
 Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 600-scale-test | 40.1 | 62.2 | 43.4 | 19.4 | 44.4 | 55.9
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test | 41.9 | 64.8 | 45.5 | 24.0 | 45.9 | 54.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms | 42.7 | 65.4 | 46.7 | 24.6 | 46.8 | 55.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms & flipping | 43.5 | 65.9 | 47.5 | 25.1 | 47.7 | 56.6 
 
 
### **3. COCOPerson results training on MSCOCO2017-trainval and testing on test-dev2017.**

 Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L|mAR@10
 :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
 **RFCN-se-air14-thin-specific** <br/> with ms-train & ohem & multigrid | 21.5 | 48.9 | 16.5 | 12.3 | 27.3 | 30.8 | 28.6
 **RFCN-resnet18-specific** <br/> with ms-train & ohem & multigrid | 38.5 | 66.1 | 39.8 | 16.8 | 47.1 | 63.0 | 41.9
 **RFCN-se-resnet50-specific** <br/> with 800-scale-train & ohem & multigrid | 39.0 | 64.1 | 41.1 | 13.5 | 48.4 | 66.4 | 43.9
 **RFCN-se-resnet50-specific** <br/> with ms-train & ohem & multigrid | 41.9 | 67.7 | 44.3 | 18.6 | 51.0 | 67.9 | 46.0
 **RFCN-se-resnet50-specific** <br/> with ms-train & ohem & multigrid & snms & flip & ms-test | 44.6 | 72.8 | 47.3 | 25.3 | 54.4 | 63.3 | 49.8
 **RFCN-se-resnet50** <br/> with ms-train & ohem & multigrid | 42.7 | 72.0 | 44.5 | 21.0 | 51.1 | 66.4 | 45.4
 **RFCN-se-inception-v2-specific** <br/> with ms-train & ohem & multigrid | 41.2 | 66.7 | 43.2 | 17.6 | 50.0 | 68.3 | 45.1
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid | 42.3 | 71.4 | 44.2 | 19.5 | 50.7 | 67.2 | 44.9
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping & ms-test | 48.0 | 79.5 | 50.0 | 28.3 | 55.8 | 67.5 | 50.8
 **RFCN-air101** <br/> with ms-train & ohem & multigrid & deformpsroi & bbox-voting & soft-nms & flipping & assign-ms-test | 54.0 | 83.9 | 58.2 | 35.2 | 61.6 | 73.0 | 55.1
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 600-scale-test | 47.7 | 76.4 | 51.1 | 25.3 | 56.8 | 70.6 | 50.2
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms & flipping | 53.4 | 82.7 | 58.0 | 33.1 | 61.8 | 73.3 | 55.0

