from caffe import layers as L
from caffe import params as P
from components import *
import caffe
from caffe.proto import caffe_pb2


class AlexNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def alexnet_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1, n.relu1, n.norm1, n.pool1 = \
            conv_relu_lrn_pool(n.data,
                               dict(num_output=96, kernel_size=11, stride=4, pad=0, group=1, weight_type='gaussian',
                                    weight_std=0.01, bias_type='constant', bias_value=0),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv2, n.relu2, n.norm2, n.pool2 = \
            conv_relu_lrn_pool(n.pool1,
                               dict(num_output=256, kernel_size=5, stride=1, pad=2, group=2, weight_type='gaussian',
                                    weight_std=0.01, bias_type='constant', bias_value=0.1),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv3, n.relu3 = \
            conv_relu(n.pool2, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=1,
                                    weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0))
        n.conv4, n.relu4 = \
            conv_relu(n.conv3, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=2,
                                    weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0.1))
        n.conv5, n.relu5, n.pool5 = \
            conv_relu_pool(n.conv4, dict(num_output=256, kernel_size=3, stride=1, pad=1, group=2,
                                         weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0.1),
                           dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, dict(num_output=4096, weight_type='gaussian', weight_std=0.05,
                                                             bias_type='constant', bias_value=0.1), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, dict(num_output=4096, weight_type='gaussian', weight_std=0.05,
                                                           bias_type='constant', bias_value=0.1), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        return n.to_proto()

    def alexnet_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1, n.bn1, n.relu1, n.pool1 = \
            conv_bn_relu_pool(n.data,
                              dict(num_output=96, kernel_size=11, stride=4, pad=0, group=1, weight_type='gaussian',
                                   weight_std=0.01, bias_type='constant', bias_value=0),
                              dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv2, n.bn2, n.relu2, n.pool2 = \
            conv_bn_relu_pool(n.pool1,
                              dict(num_output=256, kernel_size=5, stride=1, pad=2, group=2, weight_type='gaussian',
                                   weight_std=0.01, bias_type='constant', bias_value=0.1),
                              dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv3, n.bn3, n.relu3 = \
            conv_bn_relu(n.pool2, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=1,
                                       weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0))
        n.conv4, n.bn4, n.relu4 = \
            conv_bn_relu(n.bn3, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=2,
                                     weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0.1))
        n.conv5, n.bn5, n.relu5, n.pool5 = \
            conv_bn_relu_pool(n.bn4,
                              dict(num_output=256, kernel_size=3, stride=1, pad=1, group=2, weight_type='gaussian',
                                   weight_std=0.01, bias_type='constant', bias_value=0.1),
                              dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, dict(num_output=1024, weight_type='gaussian', weight_std=0.05,
                                                             bias_type='constant', bias_value=0.1), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, dict(num_output=128, weight_type='gaussian', weight_std=0.05,
                                                           bias_type='constant', bias_value=0.1), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        return n.to_proto()

    def caffenet_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1, n.relu1, n.pool1, n.norm1 = \
            conv_relu_pool_lrn(n.data,
                               dict(num_output=96, kernel_size=11, stride=4, pad=0, group=1, weight_type='gaussian',
                                    weight_std=0.01, bias_type='constant', bias_value=0),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv2, n.relu2, n.pool2, n.norm2 = \
            conv_relu_pool_lrn(n.norm1,
                               dict(num_output=256, kernel_size=5, stride=1, pad=2, group=2, weight_type='gaussian',
                                    weight_std=0.01, bias_type='constant', bias_value=1),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv3, n.relu3 = \
            conv_relu(n.norm2, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=1,
                                    weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0))
        n.conv4, n.relu4 = \
            conv_relu(n.conv3, dict(num_output=384, kernel_size=3, stride=1, pad=1, group=2,
                                    weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=1))
        n.conv5, n.relu5, n.pool5 = \
            conv_relu_pool(n.conv4, dict(num_output=256, kernel_size=3, stride=1, pad=1, group=2,
                                         weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=1),
                           dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, dict(num_output=4096, weight_type='gaussian', weight_std=0.05,
                                                             bias_type='constant', bias_value=1), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, dict(num_output=4096, weight_type='gaussian', weight_std=0.05,
                                                           bias_type='constant', bias_value=1), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        return n.to_proto()
