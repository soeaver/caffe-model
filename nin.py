from caffe import layers as L
from caffe import params as P
from components import *
import caffe
from caffe.proto import caffe_pb2


class NIN(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def nin_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1, n.relu0, n.cccp1, n.relu1, n.cccp2, n.relu2 = \
            conv_stack_3(n.data, dict(num_output=[96, 96, 96], kernel_size=[11, 1, 1], stride=[4, 1, 1],
                                      pad=[0, 0, 0], group=[1, 1, 1],
                                      weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.05, 0.05],
                                      bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool1 = L.Pooling(n.cccp2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.conv2, n.relu3, n.cccp3, n.relu4, n.cccp4, n.relu5 = \
            conv_stack_3(n.pool1, dict(num_output=[256, 256, 256], kernel_size=[5, 1, 1], stride=[1, 1, 1],
                                       pad=[2, 0, 0], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.05, 0.05, 0.05],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool2 = L.Pooling(n.cccp4, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.conv3, n.relu6, n.cccp5, n.relu7, n.cccp6, n.relu8 = \
            conv_stack_3(n.pool2, dict(num_output=[384, 384, 384], kernel_size=[3, 1, 1], stride=[2, 1, 1],
                                       pad=[1, 0, 0], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.05, 0.05],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool3 = L.Pooling(n.cccp6, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.drop7 = L.Dropout(n.pool3, in_place=True, dropout_param=dict(dropout_ratio=0.5))
        n.conv4, n.relu9, n.cccp7, n.relu10, n.cccp8, n.relu11 = \
            conv_stack_3(n.pool3, dict(num_output=[1024, 1024, 1024], kernel_size=[3, 1, 1], stride=[1, 1, 1],
                                       pad=[1, 0, 0], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.05, 0.05],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool4 = L.Pooling(n.cccp8, pool=P.Pooling.MAX, kernel_size=6, stride=1)
        if phase == 'TRAIN':
            n.loss = L.SoftmaxWithLoss(n.pool4, n.label)
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.pool4, n.label)
        return n.to_proto()

    def nin_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1, n.conv1_bn, n.relu0, n.cccp1, n.cccp1_bn, n.relu1, n.cccp2, n.cccp2_bn, n.relu2 = \
            conv_bn_stack_3(n.data, dict(num_output=[96, 96, 96], kernel_size=[11, 1, 1], stride=[4, 1, 1],
                                         pad=[0, 0, 0], group=[1, 1, 1],
                                         weight_type=['gaussian', 'gaussian', 'gaussian'],
                                         weight_std=[0.01, 0.05, 0.05],
                                         bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool1 = L.Pooling(n.cccp2_bn, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.conv2, n.conv2_bn, n.relu3, n.cccp3, n.cccp3_bn, n.relu4, n.cccp4, n.cccp4_bn, n.relu5 = \
            conv_bn_stack_3(n.pool1, dict(num_output=[256, 256, 256], kernel_size=[5, 1, 1], stride=[1, 1, 1],
                                          pad=[2, 0, 0], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.05, 0.05, 0.05],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool2 = L.Pooling(n.cccp4_bn, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.conv3, n.conv3_bn, n.relu6, n.cccp5, n.cccp5_bn, n.relu7, n.cccp6, n.cccp6_bn, n.relu8 = \
            conv_bn_stack_3(n.pool2, dict(num_output=[384, 384, 384], kernel_size=[3, 1, 1], stride=[2, 1, 1],
                                          pad=[1, 0, 0], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.01, 0.05, 0.05],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool3 = L.Pooling(n.cccp6_bn, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.drop7 = L.Dropout(n.pool3, in_place=True, dropout_param=dict(dropout_ratio=0.5))
        n.conv4, n.conv4_bn, n.relu9, n.cccp7, n.cccp7_bn, n.relu10, n.cccp8, n.cccp8_bn, n.relu11 = \
            conv_bn_stack_3(n.pool3, dict(num_output=[1024, 1024, 1024], kernel_size=[3, 1, 1], stride=[1, 1, 1],
                                          pad=[1, 0, 0], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.01, 0.05, 0.05],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool4 = L.Pooling(n.cccp8_bn, pool=P.Pooling.MAX, kernel_size=6, stride=1)
        if phase == 'TRAIN':
            n.loss = L.SoftmaxWithLoss(n.pool4, n.label)
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.pool4, n.label)
        return n.to_proto()
