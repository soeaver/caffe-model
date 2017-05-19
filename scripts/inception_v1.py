import caffe
from caffe import layers as L
from caffe import params as P


def fc_relu_drop(bottom, fc_param, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=fc_param['num_output'],
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type=fc_param['weight_type'], std=fc_param['weight_std']),
                        bias_filler=dict(type='constant', value=fc_param['bias_value']))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=1),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def inception(bottom, conv_output):
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_1x1'],
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', weight_std=1),
                             bias_filler=dict(type='constant', value=0.2))
    conv_1x1_relu = L.ReLU(conv_1x1, in_place=True)

    conv_3x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_3x3_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier', weight_std=1),
                                    bias_filler=dict(type='constant', value=0.2))
    conv_3x3_reduce_relu = L.ReLU(conv_3x3_reduce, in_place=True)
    conv_3x3 = L.Convolution(conv_3x3_reduce, kernel_size=3, num_output=conv_output['conv_3x3'], pad=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', weight_std=1),
                             bias_filler=dict(type='constant', value=0.2))
    conv_3x3_relu = L.ReLU(conv_3x3, in_place=True)

    conv_5x5_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_5x5_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier', weight_std=1),
                                    bias_filler=dict(type='constant', value=0.2))
    conv_5x5_reduce_relu = L.ReLU(conv_5x5_reduce, in_place=True)
    conv_5x5 = L.Convolution(conv_5x5_reduce, kernel_size=5, num_output=conv_output['conv_5x5'], pad=2,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', weight_std=1),
                             bias_filler=dict(type='constant', value=0.2))
    conv_5x5_relu = L.ReLU(conv_5x5, in_place=True)

    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    pool_proj = L.Convolution(pool, kernel_size=1, num_output=conv_output['pool_proj'],
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type='xavier'),
                              bias_filler=dict(type='constant', value=0.2))
    pool_proj_relu = L.ReLU(pool_proj, in_place=True)
    concat = L.Concat(conv_1x1, conv_3x3, conv_5x5, pool_proj)

    return conv_1x1, conv_1x1_relu, conv_3x3_reduce, conv_3x3_reduce_relu, conv_3x3, conv_3x3_relu, conv_5x5_reduce, \
           conv_5x5_reduce_relu, conv_5x5, conv_5x5_relu, pool, pool_proj, pool_proj_relu, concat


def inception_bn(bottom, conv_output):
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=conv_output['conv_1x1'], kernel_size=1)

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=conv_output['conv_3x3_reduce'], kernel_size=1)
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=conv_output['conv_3x3'], kernel_size=3, pad=1)

    conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_scale, conv_5x5_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=conv_output['conv_5x5_reduce'], kernel_size=1)
    conv_5x5, conv_5x5_bn, conv_5x5_scale, conv_5x5_relu = \
        factorization_conv_bn_scale_relu(conv_5x5_reduce, num_output=conv_output['conv_5x5'], kernel_size=5, pad=2)

    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    pool_proj, pool_proj_bn, pool_proj_scale, pool_proj_relu = \
        factorization_conv_bn_scale_relu(pool, num_output=conv_output['pool_proj'], kernel_size=1)

    concat = L.Concat(conv_1x1, conv_3x3, conv_5x5, pool_proj)

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_3x3_reduce, conv_3x3_reduce_bn, \
           conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, \
           conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_scale, conv_5x5_reduce_relu, conv_5x5, conv_5x5_bn, \
           conv_5x5_scale, conv_5x5_relu, pool, pool_proj, pool_proj_bn, pool_proj_scale, pool_proj_relu, concat


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

        n.conv1_7x7_s2 = L.Convolution(n.data, num_output=64, kernel_size=7, stride=2, pad=3,
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                       weight_filler=dict(type='xavier', weight_std=1),
                                       bias_filler=dict(type='constant', value=0.2))
        n.conv1_relu_7x7 = L.ReLU(n.conv1_7x7_s2, in_place=True)
        n.pool1_3x3_s2 = L.Pooling(n.conv1_7x7_s2, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
        n.pool1_norm1 = L.LRN(n.pool1_3x3_s2, local_size=5, alpha=1e-4, beta=0.75)

        n.conv2_3x3_reduce = L.Convolution(n.pool1_norm1, kernel_size=1, num_output=64, stride=1,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           weight_filler=dict(type='xavier', weight_std=1),
                                           bias_filler=dict(type='constant', value=0.2))
        n.conv2_relu_3x3_reduce = L.ReLU(n.conv2_3x3_reduce, in_place=True)

        n.conv2_3x3 = L.Convolution(n.conv2_3x3_reduce, num_output=192, kernel_size=3, stride=1, pad=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier', weight_std=1),
                                    bias_filler=dict(type='constant', value=0.2))
        n.conv2_relu_3x3 = L.ReLU(n.conv2_3x3, in_place=True)
        n.conv2_norm2 = L.LRN(n.conv2_3x3, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_3x3_s2 = L.Pooling(n.conv2_norm2, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)

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
        # loss 1
        n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss1_conv = L.Convolution(n.loss1_ave_pool, num_output=128, kernel_size=1, stride=1,
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                     weight_filler=dict(type='xavier', weight_std=1),
                                     bias_filler=dict(type='constant', value=0.2))
        n.loss1_relu_conv = L.ReLU(n.loss1_conv, in_place=True)
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
            n.loss1_accuracy_top1 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1))
            n.loss1_accuracy_top5 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

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
        # loss 2
        n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss2_conv = L.Convolution(n.loss2_ave_pool, num_output=128, kernel_size=1, stride=1,
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                     weight_filler=dict(type='xavier', weight_std=1),
                                     bias_filler=dict(type='constant', value=0.2))
        n.loss2_relu_conv = L.ReLU(n.loss2_conv, in_place=True)
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
            n.loss2_accuracy_top1 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1))
            n.loss2_accuracy_top5 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

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
            n.loss3_accuracy_top1 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1))
            n.loss3_accuracy_top5 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))
        return n.to_proto()

    def inception_bn_proto(self, batch_size, phase='TRAIN'):  # inception_bn
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1_7x7_s2, n.conv1_7x7_s2_bn, n.conv1_7x7_s2_scale, n.conv1_7x7_relu = \
            factorization_conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3)
        n.pool1_3x3_s2 = L.Pooling(n.conv1_7x7_s2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

        n.conv2_3x3_reduce, n.conv2_3x3_reduce_bn, n.conv2_3x3_reduce_scale, n.conv2_3x3_reduce_relu = \
            factorization_conv_bn_scale_relu(n.pool1_3x3_s2, num_output=64, kernel_size=1)

        n.conv2_3x3, n.conv2_3x3_bn, n.conv2_3x3_scale, n.conv2_3x3_relu = \
            factorization_conv_bn_scale_relu(n.conv2_3x3_reduce, num_output=192, kernel_size=3, pad=1)
        n.pool2_3x3_s2 = L.Pooling(n.conv2_3x3, kernel_size=3, stride=2, pool=P.Pooling.MAX)

        n.inception_3a_1x1, n.inception_3a_1x1_bn, n.inception_3a_1x1_scale, n.inception_3a_relu_1x1, \
        n.inception_3a_3x3_reduce, n.inception_3a_3x3_reduce_bn, n.inception_3a_3x3_reduce_scale, \
        n.inception_3a_relu_3x3_reduce, n.inception_3a_3x3, n.inception_3a_3x3_bn, n.inception_3a_3x3_scale, \
        n.inception_3a_relu_3x3, n.inception_3a_5x5_reduce, n.inception_3a_5x5_reduce_bn, \
        n.inception_3a_5x5_reduce_scale, n.inception_3a_relu_5x5_reduce, n.inception_3a_5x5, n.inception_3a_5x5_bn, \
        n.inception_3a_5x5_scale, n.inception_3a_relu_5x5, n.inception_3a_pool, n.inception_3a_pool_proj, \
        n.inception_3a_pool_proj_bn, n.inception_3a_pool_proj_scale, n.inception_3a_relu_pool_proj, \
        n.inception_3a_output = \
            inception_bn(n.pool2_3x3_s2, dict(conv_1x1=64, conv_3x3_reduce=96, conv_3x3=128, conv_5x5_reduce=16,
                                              conv_5x5=32, pool_proj=32))
        n.inception_3b_1x1, n.inception_3b_1x1_bn, n.inception_3b_1x1_scale, n.inception_3b_relu_1x1, \
        n.inception_3b_3x3_reduce, n.inception_3b_3x3_reduce_bn, n.inception_3b_3x3_reduce_scale, \
        n.inception_3b_relu_3x3_reduce, n.inception_3b_3x3, n.inception_3b_3x3_bn, n.inception_3b_3x3_scale, \
        n.inception_3b_relu_3x3, n.inception_3b_5x5_reduce, n.inception_3b_5x5_reduce_bn, \
        n.inception_3b_5x5_reduce_scale, n.inception_3b_relu_5x5_reduce, n.inception_3b_5x5, n.inception_3b_5x5_bn, \
        n.inception_3b_5x5_scale, n.inception_3b_relu_5x5, n.inception_3b_pool, n.inception_3b_pool_proj, \
        n.inception_3b_pool_proj_bn, n.inception_3b_pool_proj_scale, n.inception_3b_relu_pool_proj, \
        n.inception_3b_output = \
            inception_bn(n.inception_3a_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=192,
                                                     conv_5x5_reduce=32, conv_5x5=96, pool_proj=64))
        n.pool3_3x3_s2 = L.Pooling(n.inception_3b_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_4a_1x1, n.inception_4a_1x1_bn, n.inception_4a_1x1_scale, n.inception_4a_relu_1x1, \
        n.inception_4a_3x3_reduce, n.inception_4a_3x3_reduce_bn, n.inception_4a_3x3_reduce_scale, \
        n.inception_4a_relu_3x3_reduce, n.inception_4a_3x3, n.inception_4a_3x3_bn, n.inception_4a_3x3_scale, \
        n.inception_4a_relu_3x3, n.inception_4a_5x5_reduce, n.inception_4a_5x5_reduce_bn, \
        n.inception_4a_5x5_reduce_scale, n.inception_4a_relu_5x5_reduce, n.inception_4a_5x5, n.inception_4a_5x5_bn, \
        n.inception_4a_5x5_scale, n.inception_4a_relu_5x5, n.inception_4a_pool, n.inception_4a_pool_proj, \
        n.inception_4a_pool_proj_bn, n.inception_4a_pool_proj_scale, n.inception_4a_relu_pool_proj, \
        n.inception_4a_output = \
            inception_bn(n.pool3_3x3_s2, dict(conv_1x1=192, conv_3x3_reduce=96, conv_3x3=208, conv_5x5_reduce=16,
                                              conv_5x5=48, pool_proj=64))
        # loss 1
        n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss1_conv, n.loss1_conv_bn, n.loss1_conv_scale, n.loss1_relu_conv = \
            factorization_conv_bn_scale_relu(n.loss1_ave_pool, num_output=128, kernel_size=1)
        n.loss1_fc, n.loss1_relu_fc, n.loss1_drop_fc = \
            fc_relu_drop(n.loss1_conv, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                            bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
        n.loss1_classifier = L.InnerProduct(n.loss1_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss1_loss = L.SoftmaxWithLoss(n.loss1_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss1_accuracy_top1 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1))
            n.loss1_accuracy_top5 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

        n.inception_4b_1x1, n.inception_4b_1x1_bn, n.inception_4b_1x1_scale, n.inception_4b_relu_1x1, \
        n.inception_4b_3x3_reduce, n.inception_4b_3x3_reduce_bn, n.inception_4b_3x3_reduce_scale, \
        n.inception_4b_relu_3x3_reduce, n.inception_4b_3x3, n.inception_4b_3x3_bn, n.inception_4b_3x3_scale, \
        n.inception_4b_relu_3x3, n.inception_4b_5x5_reduce, n.inception_4b_5x5_reduce_bn, \
        n.inception_4b_5x5_reduce_scale, n.inception_4b_relu_5x5_reduce, n.inception_4b_5x5, n.inception_4b_5x5_bn, \
        n.inception_4b_5x5_scale, n.inception_4b_relu_5x5, n.inception_4b_pool, n.inception_4b_pool_proj, \
        n.inception_4b_pool_proj_bn, n.inception_4b_pool_proj_scale, n.inception_4b_relu_pool_proj, \
        n.inception_4b_output = \
            inception_bn(n.inception_4a_output, dict(conv_1x1=160, conv_3x3_reduce=112, conv_3x3=224,
                                                     conv_5x5_reduce=24, conv_5x5=64, pool_proj=64))
        n.inception_4c_1x1, n.inception_4c_1x1_bn, n.inception_4c_1x1_scale, n.inception_4c_relu_1x1, \
        n.inception_4c_3x3_reduce, n.inception_4c_3x3_reduce_bn, n.inception_4c_3x3_reduce_scale, \
        n.inception_4c_relu_3x3_reduce, n.inception_4c_3x3, n.inception_4c_3x3_bn, n.inception_4c_3x3_scale, \
        n.inception_4c_relu_3x3, n.inception_4c_5x5_reduce, n.inception_4c_5x5_reduce_bn, \
        n.inception_4c_5x5_reduce_scale, n.inception_4c_relu_5x5_reduce, n.inception_4c_5x5, n.inception_4c_5x5_bn, \
        n.inception_4c_5x5_scale, n.inception_4c_relu_5x5, n.inception_4c_pool, n.inception_4c_pool_proj, \
        n.inception_4c_pool_proj_bn, n.inception_4c_pool_proj_scale, n.inception_4c_relu_pool_proj, \
        n.inception_4c_output = \
            inception_bn(n.inception_4b_output, dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=256,
                                                     conv_5x5_reduce=24, conv_5x5=64, pool_proj=64))
        n.inception_4d_1x1, n.inception_4d_1x1_bn, n.inception_4d_1x1_scale, n.inception_4d_relu_1x1, \
        n.inception_4d_3x3_reduce, n.inception_4d_3x3_reduce_bn, n.inception_4d_3x3_reduce_scale, \
        n.inception_4d_relu_3x3_reduce, n.inception_4d_3x3, n.inception_4d_3x3_bn, n.inception_4d_3x3_scale, \
        n.inception_4d_relu_3x3, n.inception_4d_5x5_reduce, n.inception_4d_5x5_reduce_bn, \
        n.inception_4d_5x5_reduce_scale, n.inception_4d_relu_5x5_reduce, n.inception_4d_5x5, n.inception_4d_5x5_bn, \
        n.inception_4d_5x5_scale, n.inception_4d_relu_5x5, n.inception_4d_pool, n.inception_4d_pool_proj, \
        n.inception_4d_pool_proj_bn, n.inception_4d_pool_proj_scale, n.inception_4d_relu_pool_proj, \
        n.inception_4d_output = \
            inception_bn(n.inception_4c_output, dict(conv_1x1=112, conv_3x3_reduce=144, conv_3x3=288,
                                                     conv_5x5_reduce=32, conv_5x5=64, pool_proj=64))
        # loss 2
        n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
        n.loss2_conv, n.loss2_conv_bn, n.loss2_conv_scale, n.loss2_relu_conv = \
            factorization_conv_bn_scale_relu(n.loss2_ave_pool, num_output=128, kernel_size=1)
        n.loss2_fc, n.loss2_relu_fc, n.loss2_drop_fc = \
            fc_relu_drop(n.loss2_conv, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                               bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
        n.loss2_classifier = L.InnerProduct(n.loss2_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
        n.loss2_loss = L.SoftmaxWithLoss(n.loss2_classifier, n.label, loss_weight=0.3)
        if phase == 'TRAIN':
            pass
        else:
            n.loss2_accuracy_top1 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1))
            n.loss2_accuracy_top5 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

        n.inception_4e_1x1, n.inception_4e_1x1_bn, n.inception_4e_1x1_scale, n.inception_4e_relu_1x1, \
        n.inception_4e_3x3_reduce, n.inception_4e_3x3_reduce_bn, n.inception_4e_3x3_reduce_scale, \
        n.inception_4e_relu_3x3_reduce, n.inception_4e_3x3, n.inception_4e_3x3_bn, n.inception_4e_3x3_scale, \
        n.inception_4e_relu_3x3, n.inception_4e_5x5_reduce, n.inception_4e_5x5_reduce_bn, \
        n.inception_4e_5x5_reduce_scale, n.inception_4e_relu_5x5_reduce, n.inception_4e_5x5, n.inception_4e_5x5_bn, \
        n.inception_4e_5x5_scale, n.inception_4e_relu_5x5, n.inception_4e_pool, n.inception_4e_pool_proj, \
        n.inception_4e_pool_proj_bn, n.inception_4e_pool_proj_scale, n.inception_4e_relu_pool_proj, \
        n.inception_4e_output = \
            inception_bn(n.inception_4d_output, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
                                                     conv_5x5_reduce=32, conv_5x5=128, pool_proj=128))
        n.pool4_3x3_s2 = L.Pooling(n.inception_4e_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        n.inception_5a_1x1, n.inception_5a_1x1_bn, n.inception_5a_1x1_scale, n.inception_5a_relu_1x1, \
        n.inception_5a_3x3_reduce, n.inception_5a_3x3_reduce_bn, n.inception_5a_3x3_reduce_scale, \
        n.inception_5a_relu_3x3_reduce, n.inception_5a_3x3, n.inception_5a_3x3_bn, n.inception_5a_3x3_scale, \
        n.inception_5a_relu_3x3, n.inception_5a_5x5_reduce, n.inception_5a_5x5_reduce_bn, \
        n.inception_5a_5x5_reduce_scale, n.inception_5a_relu_5x5_reduce, n.inception_5a_5x5, n.inception_5a_5x5_bn, \
        n.inception_5a_5x5_scale, n.inception_5a_relu_5x5, n.inception_5a_pool, n.inception_5a_pool_proj, \
        n.inception_5a_pool_proj_bn, n.inception_5a_pool_proj_scale, n.inception_5a_relu_pool_proj, \
        n.inception_5a_output = \
            inception_bn(n.pool4_3x3_s2, dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
                                              conv_5x5_reduce=32, conv_5x5=128, pool_proj=128))
        n.inception_5b_1x1, n.inception_5b_1x1_bn, n.inception_5b_1x1_scale, n.inception_5b_relu_1x1, \
        n.inception_5b_3x3_reduce, n.inception_5b_3x3_reduce_bn, n.inception_5b_3x3_reduce_scale, \
        n.inception_5b_relu_3x3_reduce, n.inception_5b_3x3, n.inception_5b_3x3_bn, n.inception_5b_3x3_scale, \
        n.inception_5b_relu_3x3, n.inception_5b_5x5_reduce, n.inception_5b_5x5_reduce_bn, \
        n.inception_5b_5x5_reduce_scale, n.inception_5b_relu_5x5_reduce, n.inception_5b_5x5, n.inception_5b_5x5_bn, \
        n.inception_5b_5x5_scale, n.inception_5b_relu_5x5, n.inception_5b_pool, n.inception_5b_pool_proj, \
        n.inception_5b_pool_proj_bn, n.inception_5b_pool_proj_scale, n.inception_5b_relu_pool_proj, \
        n.inception_5b_output = \
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
            n.loss3_accuracy_top1 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1))
            n.loss3_accuracy_top5 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))
        return n.to_proto()
