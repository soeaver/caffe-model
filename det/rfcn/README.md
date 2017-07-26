## Faster RCNN
### Training faster rcnn networks on pascal voc

1.Download the network weights trained on imagenet.


2.Modify solver file
  ```
  caffe-model/det/faster_rcnn/models/pascal_voc/solver.prototxt
  ```
 - You need modify 'train_net' and 'snapshot_prefix' to the correct path or name.
  
  
3.Modify yml file
  ```
  caffe-model/det/faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml
  ```
 - The faster rcnn models will saved in '{ROOT_DIR}/output/{EXP_DIR}/{imdb.name}/' folder.


4.Training
  ```
    python train_net_multi_gpu.py --gpu_id 0,1 --solver ~/caffe-model/det/faster_rcnn/models/pascal_voc/solver.prototxt --iters 80000 --weights ~/caffe-model/cls/ilsvrc/resnet-v2/resnet101-v2/resnet101-v2_merge.caffemodel --cfg ~/caffe-model/det/faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_0712_trainval
  ```
