## Faster RCNN
### Training faster rcnn networks on pascal voc

  ```
    python train_net_multi_gpu.py --gpu_id 0,1 --solver ~/caffe-model/det/faster_rcnn/models/pascal_voc/solver.prototxt --iters 80000 --weights ~/caffe-model/cls/ilsvrc/resnet-v2/resnet101-v2/resnet101-v2_merge.caffemodel --cfg ~/caffe-model/det/faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_0712_trainval
  ```
