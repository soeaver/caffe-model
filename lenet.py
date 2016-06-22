from caffe import layers as L
from caffe import params as P
import caffe


class LeNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def lenet_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = False
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(scale=0.00390625, mirror=mirror))

        n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, stride=1,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.pool1 = L.Pooling(n.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, stride=1,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.pool2 = L.Pooling(n.conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant'))
        n.relu1 = L.ReLU(n.ip1, in_place=True)
        n.ip2 = L.InnerProduct(n.relu1, num_output=self.classifier_num,
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant'))
        n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy = L.Accuracy(n.ip2, n.label, include=dict(phase=1))

        return n.to_proto()

    def lenet_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = False
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(scale=0.00390625, mirror=mirror))

        n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, stride=1,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False)
        n.pool1 = L.Pooling(n.bn1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, stride=1,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.bn2 = L.BatchNorm(n.conv2, use_global_stats=False)
        n.pool2 = L.Pooling(n.bn2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant'))
        n.relu1 = L.ReLU(n.ip1, in_place=True)
        n.ip2 = L.InnerProduct(n.relu1, num_output=self.classifier_num,
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant'))
        n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy = L.Accuracy(n.ip2, n.label, include=dict(phase=1))

        return n.to_proto()
