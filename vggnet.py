from caffe import layers as L
from caffe import params as P
from components import *
import caffe
from caffe.proto import caffe_pb2


class VggNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def vgg_11a_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.relu1_1, n.pool1 = \
            conv_relu_pool(n.data, dict(num_output=64, kernel_size=3, stride=1, pad=1, group=1, weight_type='gaussian',
                                        weight_std=0.01, bias_type='constant', bias_value=0),
                           dict(type=P.Pooling.MAX, kernel_size=2, stride=2, pad=0))
        n.conv2_1, n.relu2_1, n.pool2 = \
            conv_relu_pool(n.pool1, dict(num_output=128, kernel_size=3, stride=1, pad=1, group=1,
                                         weight_type='gaussian', weight_std=0.01, bias_type='constant', bias_value=0),
                           dict(type=P.Pooling.MAX, kernel_size=2, stride=2, pad=0))
        n.conv3_1, n.relu3_1, n.conv3_2, n.relu3_2 = \
            conv_stack_2(n.pool2,
                         dict(num_output=[256, 256], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                              weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                              bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool3 = L.Pooling(n.conv3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv4_1, n.relu4_1, n.conv4_2, n.relu4_2 = \
            conv_stack_2(n.pool3,
                         dict(num_output=[512, 512], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                              weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                              bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool4 = L.Pooling(n.conv4_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv5_1, n.relu5_1, n.conv5_2, n.relu5_2 = \
            conv_stack_2(n.pool4,
                         dict(num_output=[512, 512], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                              weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                              bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool5 = L.Pooling(n.conv5_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.fc6, n.relu6, n.drop6 = \
            fc_relu_drop(n.pool5, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                       bias_value=0), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = \
            fc_relu_drop(n.fc6, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                     bias_value=0), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        return n.to_proto()

    def vgg_11a_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.relu1_1, n.conv1_1_bn, n.pool1 = \
            conv_bn_relu_pool(n.data,
                              dict(num_output=64, kernel_size=3, stride=1, pad=1, group=1, weight_type='gaussian',
                                   weight_std=0.01, bias_type='constant', bias_value=0),
                              dict(type=P.Pooling.MAX, kernel_size=2, stride=2, pad=0))
        n.conv2_1, n.relu2_1, n.conv2_1_bn, n.pool2 = \
            conv_bn_relu_pool(n.pool1,
                              dict(num_output=128, kernel_size=3, stride=1, pad=1, group=1, weight_type='gaussian',
                                   weight_std=0.01, bias_type='constant', bias_value=0),
                              dict(type=P.Pooling.MAX, kernel_size=2, stride=2, pad=0))
        n.conv3_1, n.conv3_1_bn, n.relu3_1, n.conv3_2, n.conv3_2, n.relu3_2 = \
            conv_bn_stack_2(n.pool2,
                            dict(num_output=[256, 256], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                                 weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                 bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool3 = L.Pooling(n.conv3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv4_1, n.conv4_1_bn, n.relu4_1, n.conv4_2, n.conv4_2_bn, n.relu4_2 = \
            conv_bn_stack_2(n.pool3,
                            dict(num_output=[512, 512], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                                 weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                 bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool4 = L.Pooling(n.conv4_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv5_1, n.conv5_1_bn, n.relu5_1, n.conv5_2, n.conv5_2_bn, n.relu5_2 = \
            conv_bn_stack_2(n.pool4,
                            dict(num_output=[512, 512], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                                 weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                 bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool5 = L.Pooling(n.conv5_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.fc6, n.relu6, n.drop6 = \
            fc_relu_drop(n.pool5, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                       bias_value=0), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = \
            fc_relu_drop(n.fc6, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                     bias_value=0), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        return n.to_proto()

    def vgg_16c_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.relu1_1, n.conv1_2, n.relu1_2 = \
            conv_stack_2(n.data, dict(num_output=[64, 64], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                                      weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                      bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv2_1, n.relu2_1, n.conv2_2, n.relu2_2 = \
            conv_stack_2(n.pool1,
                         dict(num_output=[128, 128], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1], group=[1, 1],
                              weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                              bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv3_1, n.relu3_1, n.conv3_2, n.relu3_2, n.conv3_3, n.relu3_3 = \
            conv_stack_3(n.pool2, dict(num_output=[256, 256, 256], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                       pad=[1, 1, 1], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.01, 0.01],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool3 = L.Pooling(n.conv3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv4_1, n.relu4_1, n.conv4_2, n.relu4_2, n.conv4_3, n.relu4_3 = \
            conv_stack_3(n.pool3, dict(num_output=[512, 512, 512], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                       pad=[1, 1, 1], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.01, 0.01],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool4 = L.Pooling(n.conv4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv5_1, n.relu5_1, n.conv5_2, n.relu5_2, n.conv5_3, n.relu5_3 = \
            conv_stack_3(n.pool4, dict(num_output=[512, 512, 512], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                       pad=[1, 1, 1], group=[1, 1, 1],
                                       weight_type=['gaussian', 'gaussian', 'gaussian'], weight_std=[0.01, 0.01, 0.01],
                                       bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool5 = L.Pooling(n.conv5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.fc6, n.relu6, n.drop6 = \
            fc_relu_drop(n.pool5, dict(num_output=1024, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                       bias_value=0), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = \
            fc_relu_drop(n.fc6, dict(num_output=1024, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                     bias_value=0), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        return n.to_proto()

    def vgg_16c_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.conv1_1_bn, n.relu1_1, n.conv1_2, n.conv1_2_bn, n.relu1_2 = \
            conv_bn_stack_2(n.data, dict(num_output=[64, 64], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1],
                                         group=[1, 1], weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                         bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool1 = L.Pooling(n.conv1_2_bn, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv2_1, n.conv2_1_bn, n.relu2_1, n.conv2_2, n.conv2_2_bn, n.relu2_2 = \
            conv_bn_stack_2(n.pool1, dict(num_output=[128, 128], kernel_size=[3, 3], stride=[1, 1], pad=[1, 1],
                                          group=[1, 1], weight_type=['gaussian', 'gaussian'], weight_std=[0.01, 0.01],
                                          bias_type=['constant', 'constant'], bias_value=[0, 0]))
        n.pool2 = L.Pooling(n.conv2_2_bn, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv3_1, n.conv3_1_bn, n.relu3_1, n.conv3_2, n.conv3_2_bn, n.relu3_2, n.conv3_3, n.conv3_3_bn, n.relu3_3 = \
            conv_bn_stack_3(n.pool2, dict(num_output=[256, 256, 256], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                          pad=[1, 1, 1], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.01, 0.01, 0.01],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool3 = L.Pooling(n.conv3_3_bn, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv4_1, n.conv4_1_bn, n.relu4_1, n.conv4_2, n.conv4_2_bn, n.relu4_2, n.conv4_3, n.conv4_3_bn, n.relu4_3 = \
            conv_bn_stack_3(n.pool3, dict(num_output=[512, 512, 512], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                          pad=[1, 1, 1], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.01, 0.01, 0.01],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool4 = L.Pooling(n.conv4_3_bn, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.conv5_1, n.conv5_1_bn, n.relu5_1, n.conv5_2, n.conv5_2_bn, n.relu5_2, n.conv5_3, n.conv5_3_bn, n.relu5_3 = \
            conv_bn_stack_3(n.pool4, dict(num_output=[512, 512, 512], kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                          pad=[1, 1, 1], group=[1, 1, 1],
                                          weight_type=['gaussian', 'gaussian', 'gaussian'],
                                          weight_std=[0.01, 0.01, 0.01],
                                          bias_type=['constant', 'constant', 'constant'], bias_value=[0, 0, 0]))
        n.pool5 = L.Pooling(n.conv5_3_bn, pool=P.Pooling.MAX, kernel_size=2, stride=2)
        n.fc6, n.relu6, n.drop6 = \
            fc_relu_drop(n.pool5, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                       bias_value=0), dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = \
            fc_relu_drop(n.fc6, dict(num_output=512, weight_type='gaussian', weight_std=0.01, bias_type='constant',
                                     bias_value=0), dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)
        return n.to_proto()
