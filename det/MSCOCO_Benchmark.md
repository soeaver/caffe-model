## MSCOCO Detection Benchmark

**We recommend using these caffe models with [py-RFCN-priv](https://github.com/soeaver/py-RFCN-priv)**

### **1. Results training on MSCOCO2017-trainval and testing on test-dev2017.**

 Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid | 32.6 | 53.6 | 34.5 | 12.5 | 35.1 | 48.4
 **RFCN-se-inception-v2** <br/> with ms-train & ohem & multigrid & bbox-voting & soft-nms & flipping & ms-test | 36.8 | 59.8 | 38.7 | 19.7 | 39.8 | 49.1
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
 
### **2. Context Pyramid Attention Network (CPANet) results training on MSCOCO2017-trainval and testing on test-dev2017.**
 
  Network|mAP|mAP@50|mAP@75|mAP@S|mAP@M|mAP@L
 :---:|:---:|:---:|:---:|:---:|:---:|:---:
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 600-scale-test | 40.1 | 62.2 | 43.4 | 19.4 | 44.4 | 55.9
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test | 41.9 | 64.8 | 45.5 | 24.0 | 45.9 | 54.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms | 42.7 | 65.4 | 46.7 | 24.6 | 46.8 | 55.6
 **CPANet-air101** <br/> with ms-train & ohem & multigrid & 800-scale-test & snms & flipping | 43.5 | 65.9 | 47.5 | 25.1 | 47.7 | 56.6
 
 
