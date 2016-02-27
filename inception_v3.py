from caffe import layers as L
from caffe import params as P
import caffe
from caffe.proto import caffe_pb2
from components import *


class Inception(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def inception_v3_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=299, mean_value=[104, 117, 123], mirror=mirror))
        # stage 1
        n.conv1_3x3_s2, n.conv1_3x3_s2_bn, n.conv1_3x3_relu, n.conv2_3x3_s1, n.conv2_3x3_s1_bn, n.conv2_3x3_relu, \
        n.conv3_3x3_s1, n.conv3_3x3_s1_bn, n.conv3_3x3_relu = \
            conv_bn_stack_3(n.data, dict(num_output=[32, 32, 64], kernel_size=[3, 3, 3], stride=[2, 1, 1],
                                         pad=[0, 0, 1], group=[1, 1, 1], weight_type=['xavier', 'xavier', 'xavier'],
                                         weight_std=[0.01, 0.01, 0.01],
                                         bias_type=['constant', 'constant', 'constant'], bias_value=[0.2, 0.2, 0.2]))
        n.pool1_3x3_s2 = L.Pooling(n.conv3_3x3_s1_bn, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        # stage 2
        n.conv4_3x3_reduce, n.conv4_3x3_reduce_bn, n.conv4_relu_3x3_reduce, n.conv4_3x3, n.conv4_3x3_bn, n.conv4_relu_3x3 = \
            conv_bn_stack_2(n.pool1_3x3_s2,
                            dict(num_output=[80, 192], kernel_size=[1, 3], stride=[1, 1], pad=[0, 0], group=[1, 1],
                                 weight_type=['xavier', 'xavier'], weight_std=[0.01, 0.01],
                                 bias_type=['constant', 'constant'], bias_value=[0.2, 0.2]))
        n.pool2_3x3_s2 = L.Pooling(n.conv4_3x3_bn, kernel_size=3, stride=2, pool=P.Pooling.MAX)

        # stage 3
        n.inception_3a_1x1, n.inception_3a_1x1_bn, n.inception_3a_relu_1x1, n.inception_3a_5x5_reduce, \
        n.inception_3a_5x5_reduce_bn, n.inception_3a_relu_5x5_reduce, n.inception_3a_5x5, n.inception_3a_5x5_bn, \
        n.inception_3a_relu_5x5, n.inception_3a_3x3_reduce, n.inception_3a_3x3_reduce_bn, n.inception_3a_relu_3x3_reduce, \
        n.inception_3a_3x3_1, n.inception_3a_3x3_1_bn, n.inception_3a_relu_3x3_1, n.inception_3a_3x3_2, \
        n.inception_3a_3x3_2_bn, n.inception_3a_relu_3x3_2, n.inception_3a_pool, n.inception_3a_pool_proj, \
        n.inception_3a_pool_proj_bn, n.inception_3a_relu_pool_proj, n.inception_3a_output = \
            inception_v3_7a(n.pool2_3x3_s2,
                            dict(conv_1x1=64, conv_5x5_reduce=48, conv_5x5=64, conv_3x3_reduce=64, conv_3x3_1=96,
                                 conv_3x3_2=96, pool_proj=32))
        n.inception_3b_1x1, n.inception_3b_1x1_bn, n.inception_3b_relu_1x1, n.inception_3b_5x5_reduce, \
        n.inception_3b_5x5_reduce_bn, n.inception_3b_relu_5x5_reduce, n.inception_3b_5x5, n.inception_3b_5x5_bn, \
        n.inception_3b_relu_5x5, n.inception_3b_3x3_reduce, n.inception_3b_3x3_reduce_bn, n.inception_3b_relu_3x3_reduce, \
        n.inception_3b_3x3_1, n.inception_3b_3x3_1_bn, n.inception_3b_relu_3x3_1, n.inception_3b_3x3_2, \
        n.inception_3b_3x3_2_bn, n.inception_3b_relu_3x3_2, n.inception_3b_pool, n.inception_3b_pool_proj, \
        n.inception_3b_pool_proj_bn, n.inception_3b_relu_pool_proj, n.inception_3b_output = \
            inception_v3_7a(n.inception_3a_output,
                            dict(conv_1x1=64, conv_5x5_reduce=48, conv_5x5=64, conv_3x3_reduce=64, conv_3x3_1=96,
                                 conv_3x3_2=96, pool_proj=64))
        n.inception_3c_1x1, n.inception_3c_1x1_bn, n.inception_3c_relu_1x1, n.inception_3c_5x5_reduce, \
        n.inception_3c_5x5_reduce_bn, n.inception_3c_relu_5x5_reduce, n.inception_3c_5x5, n.inception_3c_5x5_bn, \
        n.inception_3c_relu_5x5, n.inception_3c_3x3_reduce, n.inception_3c_3x3_reduce_bn, n.inception_3c_relu_3x3_reduce, \
        n.inception_3c_3x3_1, n.inception_3c_3x3_1_bn, n.inception_3c_relu_3x3_1, n.inception_3c_3x3_2, \
        n.inception_3c_3x3_2_bn, n.inception_3c_relu_3x3_2, n.inception_3c_pool, n.inception_3c_pool_proj, \
        n.inception_3c_pool_proj_bn, n.inception_3c_relu_pool_proj, n.inception_3c_output = \
            inception_v3_7a(n.inception_3b_output,
                            dict(conv_1x1=64, conv_5x5_reduce=48, conv_5x5=64, conv_3x3_reduce=64, conv_3x3_1=96,
                                 conv_3x3_2=96, pool_proj=64))
        n.inception_3d_3x3_0, n.inception_3d_3x3_0_bn, n.inception_3d_relu_3x3_0, n.inception_3d_3x3_reduce, \
        n.inception_3d_3x3_reduce_bn, n.inception_3d_relu_3x3_reduce, n.inception_3d_3x3_1, n.inception_3d_3x3_1_bn, \
        n.inception_3d_relu_3x3_1, n.inception_3d_3x3_2, n.inception_3d_3x3_2_bn, n.inception_3d_relu_3x3_2, \
        n.inception_3d_pool, n.inception_3d_output = \
            inception_v3_7b(n.inception_3c_output,
                            dict(conv_3x3_0=384, conv_3x3_reduce=64, conv_3x3_1=96, conv_3x3_2=96))

        # stage 4
        n.inception_4a_1x1, n.inception_4a_1x1_bn, n.inception_4a_relu_1x1, n.inception_4a_1x7_reduce, \
        n.inception_4a_1x7_reduce_bn, n.inception_4a_relu_1x7_reduce, n.inception_4a_1x7_0, n.inception_4a_1x7_0_bn, \
        n.inception_4a_relu_1x7_0, n.inception_4a_7x1_0, n.inception_4a_7x1_0_bn, n.inception_4a_relu_7x1_0, \
        n.inception_4a_7x1_reduce, n.inception_4a_7x1_reduce_bn, n.inception_4a_relu_7x1_reduce, \
        n.inception_4a_7x1_1, n.inception_4a_7x1_1_bn, n.inception_4a_relu_7x1_1, n.inception_4a_1x7_1, \
        n.inception_4a_1x7_1_bn, n.inception_4a_relu_1x7_1, n.inception_4a_7x1_2, n.inception_4a_7x1_2_bn, \
        n.inception_4a_relu_7x1_2, n.inception_4a_1x7_2, n.inception_4a_1x7_2_bn, n.inception_4a_relu_1x7_2, \
        n.inception_4a_pool, n.inception_4a_pool_proj, n.inception_4a_pool_proj_bn, n.inception_4a_relu_pool_proj, \
        n.inception_4a_output = \
            inception_v3_7c(n.inception_3d_output,
                            dict(conv_1x1=192, conv_1x7_reduce=128, conv_1x7_0=128, conv_7x1_0=192, conv_7x1_reduce=128,
                                 conv_1x7_1=128, conv_7x1_1=128, conv_1x7_2=128, conv_7x1_2=192, pool_proj=192))
        n.inception_4b_1x1, n.inception_4b_1x1_bn, n.inception_4b_relu_1x1, n.inception_4b_1x7_reduce, \
        n.inception_4b_1x7_reduce_bn, n.inception_4b_relu_1x7_reduce, n.inception_4b_1x7_0, n.inception_4b_1x7_0_bn, \
        n.inception_4b_relu_1x7_0, n.inception_4b_7x1_0, n.inception_4b_7x1_0_bn, n.inception_4b_relu_7x1_0, \
        n.inception_4b_7x1_reduce, n.inception_4b_7x1_reduce_bn, n.inception_4b_relu_7x1_reduce, \
        n.inception_4b_7x1_1, n.inception_4b_7x1_1_bn, n.inception_4b_relu_7x1_1, n.inception_4b_1x7_1, \
        n.inception_4b_1x7_1_bn, n.inception_4b_relu_1x7_1, n.inception_4b_7x1_2, n.inception_4b_7x1_2_bn, \
        n.inception_4b_relu_7x1_2, n.inception_4b_1x7_2, n.inception_4b_1x7_2_bn, n.inception_4b_relu_1x7_2, \
        n.inception_4b_pool, n.inception_4b_pool_proj, n.inception_4b_pool_proj_bn, n.inception_4b_relu_pool_proj, \
        n.inception_4b_output = \
            inception_v3_7c(n.inception_4a_output,
                            dict(conv_1x1=192, conv_1x7_reduce=160, conv_1x7_0=160, conv_7x1_0=192, conv_7x1_reduce=160,
                                 conv_1x7_1=160, conv_7x1_1=160, conv_1x7_2=160, conv_7x1_2=160, pool_proj=192))
        n.inception_4c_1x1, n.inception_4c_1x1_bn, n.inception_4c_relu_1x1, n.inception_4c_1x7_reduce, \
        n.inception_4c_1x7_reduce_bn, n.inception_4c_relu_1x7_reduce, n.inception_4c_1x7_0, n.inception_4c_1x7_0_bn, \
        n.inception_4c_relu_1x7_0, n.inception_4c_7x1_0, n.inception_4c_7x1_0_bn, n.inception_4c_relu_7x1_0, \
        n.inception_4c_7x1_reduce, n.inception_4c_7x1_reduce_bn, n.inception_4c_relu_7x1_reduce, \
        n.inception_4c_7x1_1, n.inception_4c_7x1_1_bn, n.inception_4c_relu_7x1_1, n.inception_4c_1x7_1, \
        n.inception_4c_1x7_1_bn, n.inception_4c_relu_1x7_1, n.inception_4c_7x1_2, n.inception_4c_7x1_2_bn, \
        n.inception_4c_relu_7x1_2, n.inception_4c_1x7_2, n.inception_4c_1x7_2_bn, n.inception_4c_relu_1x7_2, \
        n.inception_4c_pool, n.inception_4c_pool_proj, n.inception_4c_pool_proj_bn, n.inception_4c_relu_pool_proj, \
        n.inception_4c_output = \
            inception_v3_7c(n.inception_4b_output,
                            dict(conv_1x1=192, conv_1x7_reduce=160, conv_1x7_0=160, conv_7x1_0=192, conv_7x1_reduce=160,
                                 conv_1x7_1=160, conv_7x1_1=160, conv_1x7_2=160, conv_7x1_2=160, pool_proj=192))
        n.inception_4d_1x1, n.inception_4d_1x1_bn, n.inception_4d_relu_1x1, n.inception_4d_1x7_reduce, \
        n.inception_4d_1x7_reduce_bn, n.inception_4d_relu_1x7_reduce, n.inception_4d_1x7_0, n.inception_4d_1x7_0_bn, \
        n.inception_4d_relu_1x7_0, n.inception_4d_7x1_0, n.inception_4d_7x1_0_bn, n.inception_4d_relu_7x1_0, \
        n.inception_4d_7x1_reduce, n.inception_4d_7x1_reduce_bn, n.inception_4d_relu_7x1_reduce, \
        n.inception_4d_7x1_1, n.inception_4d_7x1_1_bn, n.inception_4d_relu_7x1_1, n.inception_4d_1x7_1, \
        n.inception_4d_1x7_1_bn, n.inception_4d_relu_1x7_1, n.inception_4d_7x1_2, n.inception_4d_7x1_2_bn, \
        n.inception_4d_relu_7x1_2, n.inception_4d_1x7_2, n.inception_4d_1x7_2_bn, n.inception_4d_relu_1x7_2, \
        n.inception_4d_pool, n.inception_4d_pool_proj, n.inception_4d_pool_proj_bn, n.inception_4d_relu_pool_proj, \
        n.inception_4d_output = \
            inception_v3_7c(n.inception_4c_output,
                            dict(conv_1x1=192, conv_1x7_reduce=192, conv_1x7_0=192, conv_7x1_0=192, conv_7x1_reduce=192,
                                 conv_1x7_1=192, conv_7x1_1=192, conv_1x7_2=192, conv_7x1_2=192, pool_proj=192))
        n.inception_4e_3x3_reduce, n.inception_4e_3x3_reduce_bn, n.inception_4e_relu_3x3_reduce, n.inception_4e_3x3_0, \
        n.inception_4e_3x3_0_bn, n.inception_4e_relu_3x3_0, n.inception_4e_1x7_reduce, n.inception_4e_1x7_reduce_bn, \
        n.inception_4e_relu_1x7_reduce, n.inception_4e_1x7, n.inception_4e_1x7_bn, n.inception_4e_relu_1x7, \
        n.inception_4e_7x1, n.inception_4e_7x1_bn, n.inception_4e_relu_7x1, n.inception_4e_3x3_1, \
        n.inception_4e_3x3_1_bn, n.inception_4e_relu_3x3_1, n.inception_4e_pool, n.inception_4e_output = \
            inception_v3_7d(n.inception_4d_output,
                            dict(conv_3x3_reduce=192, conv_3x3_0=320, conv_1x7_reduce=192, conv_1x7=192, conv_7x1=192,
                                 conv_3x3_1=192))

        # stage 5
        n.inception_5a_1x1, n.inception_5a_1x1_bn, n.inception_5a_relu_1x1, n.inception_5a_3x3_0_reduce, \
        n.inception_5a_3x3_0_reduce_bn, n.inception_5a_relu_3x3_0_reduce, n.inception_5a_1x3_0, n.inception_5a_1x3_0_bn, \
        n.inception_5a_relu_1x3_0, n.inception_5a_3x1_0, n.inception_5a_3x1_0_bn, n.inception_5a_relu_3x1_0, \
        n.inception_5a_3x3_1_reduce, n.inception_5a_3x3_1_reduce_bn, n.inception_5a_relu_3x3_1_reduce, n.inception_5a_3x3_1, \
        n.inception_5a_3x3_1_bn, n.inception_5a_relu_3x3_1, n.inception_5a_1x3_1, n.inception_5a_1x3_1_bn, \
        n.inception_5a_relu_1x3_1, n.inception_5a_3x1_1, n.inception_5a_3x1_1_bn, n.inception_5a_relu_3x1_1, \
        n.inception_5a_pool, n.inception_5a_pool_proj, n.inception_5a_pool_proj_bn, n.inception_5a_relu_pool_proj, \
        n.inception_5a_output = \
            inception_v3_7e(n.inception_4e_output,
                            dict(conv_1x1=320, conv_3x3_0_reduce=384, conv_1x3_0=384, conv_3x1_0=384,
                                 conv_3x3_1_reduce=448, conv_3x3_1=384, conv_1x3_1=384, conv_3x1_1=384,
                                 pooling=P.Pooling.AVE, pool_proj=192))
        n.inception_5b_1x1, n.inception_5b_1x1_bn, n.inception_5b_relu_1x1, n.inception_5b_3x3_0_reduce, \
        n.inception_5b_3x3_0_reduce_bn, n.inception_5b_relu_3x3_0_reduce, n.inception_5b_1x3_0, n.inception_5b_1x3_0_bn, \
        n.inception_5b_relu_1x3_0, n.inception_5b_3x1_0, n.inception_5b_3x1_0_bn, n.inception_5b_relu_3x1_0, \
        n.inception_5b_3x3_1_reduce, n.inception_5b_3x3_1_reduce_bn, n.inception_5b_relu_3x3_1_reduce, n.inception_5b_3x3_1, \
        n.inception_5b_3x3_1_bn, n.inception_5b_relu_3x3_1, n.inception_5b_1x3_1, n.inception_5b_1x3_1_bn, \
        n.inception_5b_relu_1x3_1, n.inception_5b_3x1_1, n.inception_5b_3x1_1_bn, n.inception_5b_relu_3x1_1, \
        n.inception_5b_pool, n.inception_5b_pool_proj, n.inception_5b_pool_proj_bn, n.inception_5b_relu_pool_proj, \
        n.inception_5b_output = \
            inception_v3_7e(n.inception_5a_output,
                            dict(conv_1x1=320, conv_3x3_0_reduce=384, conv_1x3_0=384, conv_3x1_0=384,
                                 conv_3x3_1_reduce=448, conv_3x3_1=384, conv_1x3_1=384, conv_3x1_1=384,
                                 pooling=P.Pooling.MAX, pool_proj=192))

        n.pool3_7x7_s1 = L.Pooling(n.inception_5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)
        n.classifier = L.InnerProduct(n.pool3_7x7_s1, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))

        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.loss_top1, n.loss_top5 = accuracy_top1_top5(n.classifier, n.label)
        return n.to_proto()
