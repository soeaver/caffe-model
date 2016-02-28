from caffe import layers as L
from caffe import params as P
from components import *
import caffe
from caffe.proto import caffe_pb2


class InceptionV1(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def inception_v1_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_7x7_s2, n.conv1_relu_7x7, n.pool1_3x3_s2, n.pool1_norm1 = \
            conv_relu_pool_lrn(n.data, dict(num_output=64, kernel_size=7, stride=2, pad=3, group=1,
                                            weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv2_3x3_reduce, n.conv2_relu_3x3_reduce = \
            conv_relu(n.pool1_norm1, dict(num_output=64, kernel_size=1, stride=1, pad=0, group=1,
                                          weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2))
        n.conv2_3x3, n.conv2_relu_3x3, n.conv2_norm2, n.pool2_3x3_s2 = \
            conv_relu_lrn_pool(n.conv2_3x3_reduce, dict(num_output=192, kernel_size=3, stride=1, pad=1, group=1,
                                                        weight_type='xavier', weight_std=1, bias_type='constant',
                                                        bias_value=0.2),
                               dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.inception_3a_1x1, n.inception_3a_relu_1x1, n.inception_3a_3x3_reduce, n.inception_3a_relu_3x3_reduce, \
        n.inception_3a_3x3, n.inception_3a_relu_3x3, n.inception_3a_5x5_reduce, n.inception_3a_relu_5x5_reduce, \
        n.inception_3a_5x5, n.inception_3a_relu_5x5, n.inception_3a_pool, n.inception_3a_pool_proj, \
        n.inception_3a_relu_pool_proj, n.inception_3a_output = \
            inception(n.pool2_3x3_s2, dict(conv_1x1=64, conv_3x3_reduce=96, conv_3x3=128, conv_5x5_reduce=16,
                                           conv_5x5=32, pool_proj=32))
        n.inception_3b_1x1, n.inception_3b_relu_1x1, n.inception_3b_3x3_reduce, n.inception_3b_relu_3x3_reduce, \
        n.inception_3b_3x3, n.inception_3b_relu_3x3, n.inception_3b_5x5_reduce, n.inception_3b_relu_5x5_reduce, \
        n.inception_3b_5x5, n.inception_3b_relu_5x5, n.inception_3b_pool, n.inception_3b_pool_proj, \
        n.inception_3b_relu_pool_proj, n.inception_3b_output = \
            inception(n.inception_3a_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=192, conv_5x5_reduce=32,
                                                  conv_5x5=96, pool_proj=64))
        n.pool3_3x3_s2 = L.Pooling(n.inception_3b_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_4a_1x1, n.inception_4a_relu_1x1, n.inception_4a_3x3_reduce, n.inception_4a_relu_3x3_reduce, \
        n.inception_4a_3x3, n.inception_4a_relu_3x3, n.inception_4a_5x5_reduce, n.inception_4a_relu_5x5_reduce, \
        n.inception_4a_5x5, n.inception_4a_relu_5x5, n.inception_4a_pool, n.inception_4a_pool_proj, \
        n.inception_4a_relu_pool_proj, n.inception_4a_output = \
            inception(n.pool3_3x3_s2, dict(conv_1x1=192, conv_3x3_reduce=96, conv_3x3=208, conv_5x5_reduce=16,
                                           conv_5x5=48, pool_proj=64))
        n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss1_conv, n.loss1_relu_conv = \
            conv_relu(n.loss1_ave_pool, dict(num_output=128, kernel_size=1, stride=1, pad=0, group=1,
                                             weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2))
        n.loss1_fc, n.loss1_relu_fc, n.loss1_drop_fc = \
            fc_relu_drop(n.loss1_conv, dict(num_output=1024, weight_type='xavier', weight_std=1, bias_type='constant',
                                            bias_value=0.2), dropout_ratio=0.7)
        n.loss1_classifier = L.InnerProduct(n.loss1_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss1_loss = L.SoftmaxWithLoss(n.loss1_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss1_top1, n.loss1_top5 = accuracy_top1_top5(n.loss1_classifier, n.label)

        n.inception_4b_1x1, n.inception_4b_relu_1x1, n.inception_4b_3x3_reduce, n.inception_4b_relu_3x3_reduce, \
        n.inception_4b_3x3, n.inception_4b_relu_3x3, n.inception_4b_5x5_reduce, n.inception_4b_relu_5x5_reduce, \
        n.inception_4b_5x5, n.inception_4b_relu_5x5, n.inception_4b_pool, n.inception_4b_pool_proj, \
        n.inception_4b_relu_pool_proj, n.inception_4b_output = \
            inception(n.inception_4a_output, dict(conv_1x1=160, conv_3x3_reduce=112, conv_3x3=224, conv_5x5_reduce=24,
                                                  conv_5x5=64, pool_proj=64))
        n.inception_4c_1x1, n.inception_4c_relu_1x1, n.inception_4c_3x3_reduce, n.inception_4c_relu_3x3_reduce, \
        n.inception_4c_3x3, n.inception_4c_relu_3x3, n.inception_4c_5x5_reduce, n.inception_4c_relu_5x5_reduce, \
        n.inception_4c_5x5, n.inception_4c_relu_5x5, n.inception_4c_pool, n.inception_4c_pool_proj, \
        n.inception_4c_relu_pool_proj, n.inception_4c_output = \
            inception(n.inception_4b_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=256, conv_5x5_reduce=24,
                                                  conv_5x5=64, pool_proj=64))
        n.inception_4d_1x1, n.inception_4d_relu_1x1, n.inception_4d_3x3_reduce, n.inception_4d_relu_3x3_reduce, \
        n.inception_4d_3x3, n.inception_4d_relu_3x3, n.inception_4d_5x5_reduce, n.inception_4d_relu_5x5_reduce, \
        n.inception_4d_5x5, n.inception_4d_relu_5x5, n.inception_4d_pool, n.inception_4d_pool_proj, \
        n.inception_4d_relu_pool_proj, n.inception_4d_output = \
            inception(n.inception_4c_output, dict(conv_1x1=112, conv_3x3_reduce=144, conv_3x3=288, conv_5x5_reduce=32,
                                                  conv_5x5=64, pool_proj=64))
        n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss2_conv, n.loss2_relu_conv = \
            conv_relu(n.loss2_ave_pool, dict(num_output=128, kernel_size=1, stride=1, pad=0, group=1,
                                             weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2))
        n.loss2_fc, n.loss2_relu_fc, n.loss2_drop_fc = \
            fc_relu_drop(n.loss2_conv, dict(num_output=1024, weight_type='xavier', weight_std=1, bias_type='constant',
                                            bias_value=0.2), dropout_ratio=0.7)
        n.loss2_classifier = L.InnerProduct(n.loss2_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss2_loss = L.SoftmaxWithLoss(n.loss2_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss2_top1, n.loss2_top5 = accuracy_top1_top5(n.loss2_classifier, n.label)

        n.inception_4e_1x1, n.inception_4e_relu_1x1, n.inception_4e_3x3_reduce, n.inception_4e_relu_3x3_reduce, \
        n.inception_4e_3x3, n.inception_4e_relu_3x3, n.inception_4e_5x5_reduce, n.inception_4e_relu_5x5_reduce, \
        n.inception_4e_5x5, n.inception_4e_relu_5x5, n.inception_4e_pool, n.inception_4e_pool_proj, \
        n.inception_4e_relu_pool_proj, n.inception_4e_output = \
            inception(n.inception_4d_output, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320, conv_5x5_reduce=32,
                                                  conv_5x5=128, pool_proj=128))
        n.pool4_3x3_s2 = L.Pooling(n.inception_4e_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_5a_1x1, n.inception_5a_relu_1x1, n.inception_5a_3x3_reduce, n.inception_5a_relu_3x3_reduce, \
        n.inception_5a_3x3, n.inception_5a_relu_3x3, n.inception_5a_5x5_reduce, n.inception_5a_relu_5x5_reduce, \
        n.inception_5a_5x5, n.inception_5a_relu_5x5, n.inception_5a_pool, n.inception_5a_pool_proj, \
        n.inception_5a_relu_pool_proj, n.inception_5a_output = \
            inception(n.pool4_3x3_s2, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320, conv_5x5_reduce=32,
                                           conv_5x5=128, pool_proj=128))
        n.inception_5b_1x1, n.inception_5b_relu_1x1, n.inception_5b_3x3_reduce, n.inception_5b_relu_3x3_reduce, \
        n.inception_5b_3x3, n.inception_5b_relu_3x3, n.inception_5b_5x5_reduce, n.inception_5b_relu_5x5_reduce, \
        n.inception_5b_5x5, n.inception_5b_relu_5x5, n.inception_5b_pool, n.inception_5b_pool_proj, \
        n.inception_5b_relu_pool_proj, n.inception_5b_output = \
            inception(n.inception_5a_output, dict(conv_1x1=384, conv_3x3_reduce=192, conv_3x3=384, conv_5x5_reduce=48,
                                                  conv_5x5=128, pool_proj=128))
        n.pool5_7x7_s1 = L.Pooling(n.inception_5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)
        n.pool5_drop_7x7_s1 = L.Dropout(n.pool5_7x7_s1, in_place=True,
                                        dropout_param=dict(dropout_ratio=0.4))
        n.loss3_classifier = L.InnerProduct(n.pool5_7x7_s1, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss3_loss = L.SoftmaxWithLoss(n.loss3_classifier, n.label, loss_weight=1)
        if phase == 'TRAIN':
            pass
        else:
            n.loss3_top1, n.loss3_top5 = accuracy_top1_top5(n.loss2_classifier, n.label)
        return n.to_proto()

    def inception_v1_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_7x7_s2, n.conv1_7x7_s2_bn, n.conv1_relu_7x7, n.pool1_3x3_s2 = \
            conv_bn_relu_pool(n.data, dict(num_output=64, kernel_size=7, stride=2, pad=3, group=1,
                                           weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2),
                              dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
        n.conv2_3x3_reduce, n.conv2_3x3_reduce_bn, n.conv2_relu_3x3_reduce = \
            conv_bn_relu(n.pool1_3x3_s2, dict(num_output=64, kernel_size=1, stride=1, pad=0, group=1,
                                              weight_type='xavier', weight_std=1, bias_type='constant', bias_value=0.2))
        n.conv2_3x3, n.conv2_3x3_bn, n.conv2_relu_3x3, n.pool2_3x3_s2 = \
            conv_bn_relu_pool(n.conv2_3x3_reduce_bn, dict(num_output=192, kernel_size=3, stride=1, pad=1, group=1,
                                                          weight_type='xavier', weight_std=1, bias_type='constant',
                                                          bias_value=0.2),
                              dict(type=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))

        n.inception_3a_1x1, n.inception_3a_1x1_bn, n.inception_3a_relu_1x1, n.inception_3a_3x3_reduce, \
        n.inception_3a_3x3_reduce_bn, n.inception_3a_relu_3x3_reduce, n.inception_3a_3x3, n.inception_3a_3x3_bn, \
        n.inception_3a_relu_3x3, n.inception_3a_5x5_reduce, n.inception_3a_5x5_reduce_bn, n.inception_3a_relu_5x5_reduce, \
        n.inception_3a_5x5, n.inception_3a_5x5_bn, n.inception_3a_relu_5x5, n.inception_3a_pool, n.inception_3a_pool_proj, \
        n.inception_3a_pool_proj_bn, n.inception_3a_relu_pool_proj, n.inception_3a_output = \
            inception_bn(n.pool2_3x3_s2, dict(conv_1x1=64, conv_3x3_reduce=96, conv_3x3=128, conv_5x5_reduce=16,
                                              conv_5x5=32, pool_proj=32))
        n.inception_3b_1x1, n.inception_3b_1x1_bn, n.inception_3b_relu_1x1, n.inception_3b_3x3_reduce, \
        n.inception_3b_3x3_reduce_bn, n.inception_3b_relu_3x3_reduce, n.inception_3b_3x3, n.inception_3b_3x3_bn, \
        n.inception_3b_relu_3x3, n.inception_3b_5x5_reduce, n.inception_3b_5x5_reduce_bn, n.inception_3b_relu_5x5_reduce, \
        n.inception_3b_5x5, n.inception_3b_5x5_bn, n.inception_3b_relu_5x5, n.inception_3b_pool, n.inception_3b_pool_proj, \
        n.inception_3b_pool_proj_bn, n.inception_3b_relu_pool_proj, n.inception_3b_output = \
            inception_bn(n.inception_3a_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=192,
                                                     conv_5x5_reduce=32, conv_5x5=96, pool_proj=64))
        n.pool3_3x3_s2 = L.Pooling(n.inception_3b_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_4a_1x1, n.inception_4a_1x1_bn, n.inception_4a_relu_1x1, n.inception_4a_3x3_reduce, \
        n.inception_4a_3x3_reduce_bn, n.inception_4a_relu_3x3_reduce, n.inception_4a_3x3, n.inception_4a_3x3_bn, \
        n.inception_4a_relu_3x3, n.inception_4a_5x5_reduce, n.inception_4a_5x5_reduce_bn, n.inception_4a_relu_5x5_reduce, \
        n.inception_4a_5x5, n.inception_4a_5x5_bn, n.inception_4a_relu_5x5, n.inception_4a_pool, n.inception_4a_pool_proj, \
        n.inception_4a_pool_proj_bn, n.inception_4a_relu_pool_proj, n.inception_4a_output = \
            inception_bn(n.pool3_3x3_s2, dict(conv_1x1=192, conv_3x3_reduce=96, conv_3x3=208, conv_5x5_reduce=16,
                                              conv_5x5=48, pool_proj=64))
        n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss1_conv, n.loss1_conv_bn, n.loss1_relu_conv = \
            conv_bn_relu(n.loss1_ave_pool, dict(num_output=128, kernel_size=1, stride=1, pad=0, group=1,
                                                weight_type='xavier', weight_std=1, bias_type='constant',
                                                bias_value=0.2))
        n.loss1_fc, n.loss1_relu_fc, n.loss1_drop_fc = \
            fc_relu_drop(n.loss1_conv_bn, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                               bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
        n.loss1_classifier = L.InnerProduct(n.loss1_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss1_loss = L.SoftmaxWithLoss(n.loss1_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss1_top1, n.loss1_top5 = accuracy_top1_top5(n.loss1_classifier, n.label)

        n.inception_4b_1x1, n.inception_4b_1x1_bn, n.inception_4b_relu_1x1, n.inception_4b_3x3_reduce, \
        n.inception_4b_3x3_reduce_bn, n.inception_4b_relu_3x3_reduce, n.inception_4b_3x3, n.inception_4b_3x3_bn, \
        n.inception_4b_relu_3x3, n.inception_4b_5x5_reduce, n.inception_4b_5x5_reduce_bn, n.inception_4b_relu_5x5_reduce, \
        n.inception_4b_5x5, n.inception_4b_5x5_bn, n.inception_4b_relu_5x5, n.inception_4b_pool, n.inception_4b_pool_proj, \
        n.inception_4b_pool_proj_bn, n.inception_4b_relu_pool_proj, n.inception_4b_output = \
            inception_bn(n.inception_4a_output, dict(conv_1x1=160, conv_3x3_reduce=112, conv_3x3=224,
                                                     conv_5x5_reduce=24, conv_5x5=64, pool_proj=64))
        n.inception_4c_1x1, n.inception_4c_1x1_bn, n.inception_4c_relu_1x1, n.inception_4c_3x3_reduce, \
        n.inception_4c_3x3_reduce_bn, n.inception_4c_relu_3x3_reduce, n.inception_4c_3x3, n.inception_4c_3x3_bn, \
        n.inception_4c_relu_3x3, n.inception_4c_5x5_reduce, n.inception_4c_5x5_reduce_nm, n.inception_4c_relu_5x5_reduce, \
        n.inception_4c_5x5, n.inception_4c_5x5_bn, n.inception_4c_relu_5x5, n.inception_4c_pool, n.inception_4c_pool_proj, \
        n.inception_4c_pool_proj_bn, n.inception_4c_relu_pool_proj, n.inception_4c_output = \
            inception_bn(n.inception_4b_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=256,
                                                     conv_5x5_reduce=24, conv_5x5=64, pool_proj=64))
        n.inception_4d_1x1, n.inception_4d_1x1_bn, n.inception_4d_relu_1x1, n.inception_4d_3x3_reduce, \
        n.inception_4d_3x3_reduce_bn, n.inception_4d_relu_3x3_reduce, n.inception_4d_3x3, n.inception_4d_3x3_bn, \
        n.inception_4d_relu_3x3, n.inception_4d_5x5_reduce, n.inception_4d_5x5_reduce_bn, n.inception_4d_relu_5x5_reduce, \
        n.inception_4d_5x5, n.inception_4d_5x5_bn, n.inception_4d_relu_5x5, n.inception_4d_pool, n.inception_4d_pool_proj, \
        n.inception_4d_pool_proj_bn, n.inception_4d_relu_pool_proj, n.inception_4d_output = \
            inception_bn(n.inception_4c_output, dict(conv_1x1=112, conv_3x3_reduce=144, conv_3x3=288,
                                                     conv_5x5_reduce=32, conv_5x5=64, pool_proj=64))
        n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss2_conv, n.loss2_conv_bn, n.loss2_relu_conv = \
            conv_bn_relu(n.loss2_ave_pool, dict(num_output=128, kernel_size=1, stride=1, pad=0, group=1,
                                                weight_type='xavier', weight_std=1, bias_type='constant',
                                                bias_value=0.2))
        n.loss2_fc, n.loss2_relu_fc, n.loss2_drop_fc = \
            fc_relu_drop(n.loss2_conv_bn, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                               bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
        n.loss2_classifier = L.InnerProduct(n.loss2_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss2_loss = L.SoftmaxWithLoss(n.loss2_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss2_top1, n.loss2_top5 = accuracy_top1_top5(n.loss2_classifier, n.label)

        n.inception_4e_1x1, n.inception_4e_1x1_bn, n.inception_4e_relu_1x1, n.inception_4e_3x3_reduce, \
        n.inception_4e_3x3_reduce_bn, n.inception_4e_relu_3x3_reduce, n.inception_4e_3x3, n.inception_4e_3x3_bn, \
        n.inception_4e_relu_3x3, n.inception_4e_5x5_reduce, n.inception_4e_5x5_reduce_bn, n.inception_4e_relu_5x5_reduce, \
        n.inception_4e_5x5, n.inception_4e_5x5_bn, n.inception_4e_relu_5x5, n.inception_4e_pool, n.inception_4e_pool_proj, \
        n.inception_4e_pool_proj_bn, n.inception_4e_relu_pool_proj, n.inception_4e_output = \
            inception_bn(n.inception_4d_output, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
                                                     conv_5x5_reduce=32, conv_5x5=128, pool_proj=128))
        n.pool4_3x3_s2 = L.Pooling(n.inception_4e_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_5a_1x1, n.inception_5a_1x1_bn, n.inception_5a_relu_1x1, n.inception_5a_3x3_reduce, \
        n.inception_5a_3x3_reduce_bn, n.inception_5a_relu_3x3_reduce, n.inception_5a_3x3, n.inception_5a_3x3_bn, \
        n.inception_5a_relu_3x3, n.inception_5a_5x5_reduce, n.inception_5a_5x5_reduce_bn, n.inception_5a_relu_5x5_reduce, \
        n.inception_5a_5x5, n.inception_5a_5x5_bn, n.inception_5a_relu_5x5, n.inception_5a_pool, n.inception_5a_pool_proj, \
        n.inception_5a_pool_proj_bn, n.inception_5a_relu_pool_proj, n.inception_5a_output = \
            inception_bn(n.pool4_3x3_s2, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
                                              conv_5x5_reduce=32, conv_5x5=128, pool_proj=128))
        n.inception_5b_1x1, n.inception_5b_1x1_bn, n.inception_5b_relu_1x1, n.inception_5b_3x3_reduce, \
        n.inception_5b_3x3_reduce_bn, n.inception_5b_relu_3x3_reduce, n.inception_5b_3x3, n.inception_5b_3x3_bn, \
        n.inception_5b_relu_3x3, n.inception_5b_5x5_reduce, n.inception_5b_5x5_reduce_bn, n.inception_5b_relu_5x5_reduce, \
        n.inception_5b_5x5, n.inception_5b_5x5_bn, n.inception_5b_relu_5x5, n.inception_5b_pool, n.inception_5b_pool_proj, \
        n.inception_5b_pool_proj_bn, n.inception_5b_relu_pool_proj, n.inception_5b_output = \
            inception_bn(n.inception_5a_output, dict(conv_1x1=384, conv_3x3_reduce=192, conv_3x3=384,
                                                     conv_5x5_reduce=48, conv_5x5=128, pool_proj=128))
        n.pool5_7x7_s1 = L.Pooling(n.inception_5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)
        n.pool5_drop_7x7_s1 = L.Dropout(n.pool5_7x7_s1, in_place=True,
                                        dropout_param=dict(dropout_ratio=0.4))
        n.loss3_classifier = L.InnerProduct(n.pool5_7x7_s1, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss3_loss = L.SoftmaxWithLoss(n.loss3_classifier, n.label, loss_weight=1)
        if phase == 'TRAIN':
            pass
        else:
            n.loss3_top1, n.loss3_top5 = accuracy_top1_top5(n.loss2_classifier, n.label)
        return n.to_proto()
