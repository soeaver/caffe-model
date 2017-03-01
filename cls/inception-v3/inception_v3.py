import caffe
from caffe import layers as L
from caffe import params as P


def fc_relu_drop(bottom, num_output=1024, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=num_output,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='xavier', std=1),
                        bias_filler=dict(type='constant', value=0.2))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def factorization_conv_mxn(bottom, num_output=64, kernel_h=1, kernel_w=7, stride=1, pad_h=3, pad_w=0):
    conv_mxn = L.Convolution(bottom, num_output=num_output, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride,
                             pad_h=pad_h, pad_w=pad_w,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', std=0.01),
                             bias_filler=dict(type='constant', value=0.2))
    conv_mxn_bn = L.BatchNorm(conv_mxn, use_global_stats=False, in_place=True)
    conv_mxn_scale = L.Scale(conv_mxn, scale_param=dict(bias_term=True), in_place=True)
    conv_mxn_relu = L.ReLU(conv_mxn, in_place=True)

    return conv_mxn, conv_mxn_bn, conv_mxn_scale, conv_mxn_relu


def stem_v3_299x299(bottom):
    """
    input:3x299x299
    output:192x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=3, stride=2)  # 32x149x149
    conv2_3x3_s1, conv2_3x3_s1_bn, conv2_3x3_s1_scale, conv2_3x3_s1_relu = \
        factorization_conv_bn_scale_relu(conv1_3x3_s2, num_output=32, kernel_size=3)  # 32x147x147
    conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu = \
        factorization_conv_bn_scale_relu(conv2_3x3_s1, num_output=64, kernel_size=3, pad=1)  # 64x147x147
    pool1_3x3_s2 = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x73x73

    conv4_3x3_reduce, conv4_3x3_reduce_bn, conv4_3x3_reduce_scale, conv4_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(pool1_3x3_s2, num_output=80, kernel_size=1)  # 80x73x73
    conv4_3x3, conv4_3x3_bn, conv4_3x3_scale, conv4_3x3_relu = \
        factorization_conv_bn_scale_relu(conv4_3x3_reduce, num_output=192, kernel_size=3)  # 192x71x71
    pool2_3x3_s2 = L.Pooling(conv4_3x3, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 192x35x35

    return conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_relu, conv2_3x3_s1, conv2_3x3_s1_bn, \
           conv2_3x3_s1_scale, conv2_3x3_s1_relu, conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu, \
           pool1_3x3_s2, conv4_3x3_reduce, conv4_3x3_reduce_bn, conv4_3x3_reduce_scale, conv4_3x3_reduce_relu, \
           conv4_3x3, conv4_3x3_bn, conv4_3x3_scale, conv4_3x3_relu, pool2_3x3_s2


def inception_v3_a(bottom, pool_proj_num_output=32):
    """
    input:192or256or288x35x35
    output:256or288x35x35
    :param pool_proj_num_output: num_output of pool_proj
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=1)  # 64x35x35

    conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_scale, conv_5x5_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=48, kernel_size=1)  # 48x35x35
    conv_5x5, conv_5x5_bn, conv_5x5_scale, conv_5x5_relu = \
        factorization_conv_bn_scale_relu(conv_5x5_reduce, num_output=64, kernel_size=5, pad=2)  # 64x35x35

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, kernel_size=1, num_output=64)  # 64x35x35
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, kernel_size=3, num_output=96, pad=1)  # 96x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3, kernel_size=3, num_output=96, pad=1)  # 96x35x35

    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 192x35x35
    pool_proj, pool_proj_bn, pool_proj_scale, pool_proj_relu = \
        factorization_conv_bn_scale_relu(pool, kernel_size=1, num_output=pool_proj_num_output)  # 32x35x35

    concat = L.Concat(conv_1x1, conv_5x5, conv_3x3_2, pool_proj)  # 256or288(64+64+96+32or64)x35x35

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_5x5_reduce, conv_5x5_reduce_bn, \
           conv_5x5_reduce_scale, conv_5x5_reduce_relu, conv_5x5, conv_5x5_bn, conv_5x5_scale, conv_5x5_relu, \
           conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, \
           conv_3x3_scale, conv_3x3_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu, pool, \
           pool_proj, pool_proj_bn, pool_proj_scale, pool_proj_relu, concat


def reduction_v3_a(bottom):
    """
    input:288x35x35
    output:768x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 384x17x17

    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(bottom, kernel_size=3, num_output=384, stride=2)  # 384x17x17

    conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=1)  # 64x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2_reduce, num_output=96, kernel_size=3, pad=1)  # 96x35x35
    conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2, num_output=96, kernel_size=3, stride=2)  # 96x17x17

    concat = L.Concat(pool, conv_3x3, conv_3x3_3)  # 768(288+384+96)x17x17

    return pool, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, concat


def inception_v3_b(bottom, outs=128):
    """
    input:768x17x17
    output:768x17x17
    :param outs: num_outputs
    :param bottom: bottom layer
    :return: layers
    """
    pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 768x17x17
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(pool_ave, num_output=192, kernel_size=1)  # 192x17x17

    conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, conv_1x1_2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x17x17

    conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_scale, conv_1x7_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=outs, kernel_size=1)  # outsx17x17
    conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu = \
        factorization_conv_mxn(conv_1x7_reduce, num_output=outs, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # outsx17x17
    conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu = \
        factorization_conv_mxn(conv_1x7, num_output=192, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 192x17x17

    conv_7x1_reduce, conv_7x1_reduce_bn, conv_7x1_reduce_scale, conv_7x1_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=outs, kernel_size=1)  # outsx17x17
    conv_7x1_2, conv_7x1_2_bn, conv_7x1_2_scale, conv_7x1_2_relu = \
        factorization_conv_mxn(conv_7x1_reduce, num_output=outs, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # outsx17x17
    conv_1x7_2, conv_1x7_2_bn, conv_1x7_2_scale, conv_1x7_2_relu = \
        factorization_conv_mxn(conv_7x1_2, num_output=outs, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # outsx17x17
    conv_7x1_3, conv_7x1_3_bn, conv_7x1_3_scale, conv_7x1_3_relu = \
        factorization_conv_mxn(conv_1x7_2, num_output=outs, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # outsx17x17
    conv_1x7_3, conv_1x7_3_bn, conv_1x7_3_scale, conv_1x7_3_relu = \
        factorization_conv_mxn(conv_7x1_3, num_output=192, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 192x17x17

    concat = L.Concat(conv_1x1_2, conv_7x1, conv_1x7_3, conv_1x1)  # 768(192+192+192+192)x17x17

    return pool_ave, conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, \
           conv_1x1_2_relu, conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_scale, conv_1x7_reduce_relu, \
           conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu, conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu, \
           conv_7x1_reduce, conv_7x1_reduce_bn, conv_7x1_reduce_scale, conv_7x1_reduce_relu, conv_7x1_2, conv_7x1_2_bn, \
           conv_7x1_2_scale, conv_7x1_2_relu, conv_1x7_2, conv_1x7_2_bn, conv_1x7_2_scale, conv_1x7_2_relu, conv_7x1_3, \
           conv_7x1_3_bn, conv_7x1_3_scale, conv_7x1_3_relu, conv_1x7_3, conv_1x7_3_bn, conv_1x7_3_scale, conv_1x7_3_relu, \
           concat


def reduction_v3_b(bottom):
    """
    input:768x17x17
    output:1280x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 768x8x8

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x17x17
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=320, kernel_size=3, stride=2)  # 320x8x8

    conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_scale, conv_1x7_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x17x17
    conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu = \
        factorization_conv_mxn(conv_1x7_reduce, num_output=192, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 192x17x17
    conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu = \
        factorization_conv_mxn(conv_1x7, num_output=192, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 192x17x17
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_7x1, num_output=192, kernel_size=3, stride=2)  # 192x8x8

    concat = L.Concat(pool, conv_3x3, conv_3x3_2)  # 1280(768+320+192)x8x8

    return pool, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, \
           conv_3x3_scale, conv_3x3_relu, conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_scale, conv_1x7_reduce_relu, \
           conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu, conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu, \
           conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu, concat


def inception_v3_c(bottom, pool=P.Pooling.AVE):
    """
    input:1280or2048x8x8
    output:2048x8x8
    :param pool: pool_type
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=pool)  # 1280or2048x8x8
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(pool, num_output=192, kernel_size=1)  # 192x8x8

    conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, conv_1x1_2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=320, kernel_size=1)  # 320x8x8

    conv_1x3_reduce, conv_1x3_reduce_bn, conv_1x3_reduce_scale, conv_1x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=384, kernel_size=1)  # 384x8x8
    conv_1x3, conv_1x3_bn, conv_1x3_scale, conv_1x3_relu = \
        factorization_conv_mxn(conv_1x3_reduce, num_output=384, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 384x8x8
    conv_3x1, conv_3x1_bn, conv_3x1_scale, conv_3x1_relu = \
        factorization_conv_mxn(conv_1x3_reduce, num_output=384, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 384x8x8

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=448, kernel_size=1)  # 448x8x8
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=384, kernel_size=3, pad=1)  # 384x8x8
    conv_1x3_2, conv_1x3_2_bn, conv_1x3_2_scale, conv_1x3_2_relu = \
        factorization_conv_mxn(conv_3x3, num_output=384, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 384x8x8
    conv_3x1_2, conv_3x1_2_bn, conv_3x1_2_scale, conv_3x1_2_relu = \
        factorization_conv_mxn(conv_3x3, num_output=384, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 384x8x8

    concat = L.Concat(conv_1x1_2, conv_1x3, conv_3x1, conv_1x3_2, conv_3x1_2, conv_1x1)  # 2048(192+320+384+384+384+384)x8x8

    return pool, conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, \
           conv_1x1_2_relu, conv_1x3_reduce, conv_1x3_reduce_bn, conv_1x3_reduce_scale, conv_1x3_reduce_relu, conv_1x3, \
           conv_1x3_bn, conv_1x3_scale, conv_1x3_relu, conv_3x1, conv_3x1_bn, conv_3x1_scale, conv_3x1_relu, \
           conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, \
           conv_3x3_scale, conv_3x3_relu, conv_1x3_2, conv_1x3_2_bn, conv_1x3_2_scale, conv_1x3_2_relu, conv_3x1_2, \
           conv_3x1_2_bn, conv_3x1_2_scale, conv_3x1_2_relu, concat


class InceptionV3(object):
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

        # stem 
        n.conv1_3x3_s2, n.conv1_3x3_s2_bn, n.conv1_3x3_s2_scale, n.conv1_3x3_relu, n.conv2_3x3_s1, n.conv2_3x3_s1_bn, \
        n.conv2_3x3_s1_scale, n.conv2_3x3_relu, n.conv3_3x3_s1, n.conv3_3x3_s1_bn, n.conv3_3x3_s1_scale, n.conv3_3x3_relu, \
        n.pool1_3x3_s2, n.conv4_3x3_reduce, n.conv4_3x3_reduce_bn, n.conv4_relu_3x3_reduce, n.conv4_3x3_reduce_scale, \
        n.conv4_3x3, n.conv4_3x3_bn, n.conv4_3x3_scale, n.conv4_relu_3x3, n.pool2_3x3_s2 = \
            stem_v3_299x299(n.data)  # 192x35x35

        # 3 x inception_v3_a
        n.inception_a1_1x1, n.inception_a1_1x1_bn, n.inception_a1_1x1_scale, n.inception_a1_1x1_relu, n.inception_a1_5x5_reduce, \
        n.inception_a1_5x5_reduce_bn, n.inception_a1_5x5_reduce_scale, n.inception_a1_5x5_reduce_relu, n.inception_a1_5x5, \
        n.inception_a1_5x5_bn, n.inception_a1_5x5_scale, n.inception_a1_5x5_relu, n.inception_a1_3x3_reduce, \
        n.inception_a1_3x3_reduce_bn, n.inception_a1_3x3_reduce_scale, n.inception_a1_3x3_reduce_relu, n.inception_a1_3x3_1, \
        n.inception_a1_3x3_bn, n.inception_a1_3x3_scale, n.inception_a1_3x3_relu, n.inception_a1_3x3_2, \
        n.inception_a1_3x3_2_bn, n.inception_a1_3x3_2_scale, n.inception_a1_3x3_2_relu, n.inception_a1_pool, \
        n.inception_a1_pool_proj, n.inception_a1_pool_proj_bn, n.inception_a1_pool_proj_scale, n.inception_a1_pool_proj_relu, \
        n.inception_a1_output = \
            inception_v3_a(n.pool2_3x3_s2)  # 256x35x35
        n.inception_a2_1x1, n.inception_a2_1x1_bn, n.inception_a2_1x1_scale, n.inception_a2_1x1_relu, n.inception_a2_5x5_reduce, \
        n.inception_a2_5x5_reduce_bn, n.inception_a2_5x5_reduce_scale, n.inception_a2_5x5_reduce_relu, n.inception_a2_5x5, \
        n.inception_a2_5x5_bn, n.inception_a2_5x5_scale, n.inception_a2_5x5_relu, n.inception_a2_3x3_reduce, \
        n.inception_a2_3x3_reduce_bn, n.inception_a2_3x3_reduce_scale, n.inception_a2_3x3_reduce_relu, n.inception_a2_3x3_1, \
        n.inception_a2_3x3_bn, n.inception_a2_3x3_scale, n.inception_a2_3x3_relu, n.inception_a2_3x3_2, \
        n.inception_a2_3x3_2_bn, n.inception_a2_3x3_2_scale, n.inception_a2_3x3_2_relu, n.inception_a2_pool, \
        n.inception_a2_pool_proj, n.inception_a2_pool_proj_bn, n.inception_a2_pool_proj_scale, n.inception_a2_pool_proj_relu, \
        n.inception_a2_output = \
            inception_v3_a(n.inception_a1_output, pool_proj_num_output=64)  # 288x35x35
        n.inception_a3_1x1, n.inception_a3_1x1_bn, n.inception_a3_1x1_scale, n.inception_a3_1x1_relu, n.inception_a3_5x5_reduce, \
        n.inception_a3_5x5_reduce_bn, n.inception_a3_5x5_reduce_scale, n.inception_a3_5x5_reduce_relu, n.inception_a3_5x5, \
        n.inception_a3_5x5_bn, n.inception_a3_5x5_scale, n.inception_a3_5x5_relu, n.inception_a3_3x3_reduce, \
        n.inception_a3_3x3_reduce_bn, n.inception_a3_3x3_reduce_scale, n.inception_a3_3x3_reduce_relu, n.inception_a3_3x3_1, \
        n.inception_a3_3x3_bn, n.inception_a3_3x3_scale, n.inception_a3_3x3_relu, n.inception_a3_3x3_2, \
        n.inception_a3_3x3_2_bn, n.inception_a3_3x3_2_scale, n.inception_a3_3x3_2_relu, n.inception_a3_pool, \
        n.inception_a3_pool_proj, n.inception_a3_pool_proj_bn, n.inception_a3_pool_proj_scale, n.inception_a3_pool_proj_relu, \
        n.inception_a3_output = \
            inception_v3_a(n.inception_a2_output, pool_proj_num_output=64)  # 288x35x35

        # reduction_v3_a
        n.reduction_a_pool, n.reduction_a_3x3, n.reduction_a_3x3_bn, n.reduction_a_3x3_scale, n.reduction_a_3x3_relu, \
        n.reduction_a_3x3_2_reduce, n.reduction_a_3x3_2_reduce_bn, n.reduction_a_3x3_2_reduce_scale, n.reduction_a_3x3_2_reduce_relu, \
        n.reduction_a_3x3_2, n.reduction_a_3x3_2_bn, n.reduction_a_3x3_2_scale, n.reduction_a_3x3_2_relu, n.reduction_a_3x3_3, \
        n.reduction_a_3x3_3_bn, n.reduction_a_3x3_3_scale, n.reduction_a_3x3_3_relu, n.reduction_a_concat = \
            reduction_v3_a(n.inception_a3_output)  # 768x17x17

        # 4 x inception_v3_b
        n.inception_b1_pool_ave, n.inception_b1_1x1, n.inception_b1_1x1_bn, n.inception_b1_1x1_scale, n.inception_b1_1x1_relu, \
        n.inception_b1_1x1_2, n.inception_b1_1x1_2_bn, n.inception_b1_1x1_2_scale, n.inception_b1_1x1_2_relu, \
        n.inception_b1_1x7_reduce, n.inception_b1_1x7_reduce_bn, n.inception_b1_1x7_reduce_scale, n.inception_b1_1x7_reduce_relu, \
        n.inception_b1_1x7, n.inception_b1_1x7_bn, n.inception_b1_1x7_scale, n.inception_b1_1x7_relu, n.inception_b1_7x1, \
        n.inception_b1_7x1_bn, n.inception_b1_7x1_scale, n.inception_b1_7x1_relu, n.inception_b1_7x1_reduce, n.inception_b1_7x1_reduce_bn, \
        n.inception_b1_7x1_reduce_scale, n.inception_b1_7x1_reduce_relu, n.inception_b1_7x1_2, n.inception_b1_7x1_2_bn, \
        n.inception_b1_7x1_2_scale, n.inception_b1_7x1_2_relu, n.inception_b1_1x7_2, n.inception_b1_1x7_2_bn, n.inception_b1_1x7_2_scale, \
        n.inception_b1_1x7_2_relu, n.inception_b1_7x1_3, n.inception_b1_7x1_3_bn, n.inception_b1_7x1_3_scale, n.inception_b1_7x1_3_relu, \
        n.inception_b1_1x7_3, n.inception_b1_1x7_3_bn, n.inception_b1_1x7_3_scale, n.inception_b1_1x7_3_relu, n.inception_b1_concat = \
            inception_v3_b(n.reduction_a_concat, outs=128)  # 768x17x17
        n.inception_b2_pool_ave, n.inception_b2_1x1, n.inception_b2_1x1_bn, n.inception_b2_1x1_scale, n.inception_b2_1x1_relu, \
        n.inception_b2_1x1_2, n.inception_b2_1x1_2_bn, n.inception_b2_1x1_2_scale, n.inception_b2_1x1_2_relu, \
        n.inception_b2_1x7_reduce, n.inception_b2_1x7_reduce_bn, n.inception_b2_1x7_reduce_scale, n.inception_b2_1x7_reduce_relu, \
        n.inception_b2_1x7, n.inception_b2_1x7_bn, n.inception_b2_1x7_scale, n.inception_b2_1x7_relu, n.inception_b2_7x1, \
        n.inception_b2_7x1_bn, n.inception_b2_7x1_scale, n.inception_b2_7x1_relu, n.inception_b2_7x1_reduce, n.inception_b2_7x1_reduce_bn, \
        n.inception_b2_7x1_reduce_scale, n.inception_b2_7x1_reduce_relu, n.inception_b2_7x1_2, n.inception_b2_7x1_2_bn, \
        n.inception_b2_7x1_2_scale, n.inception_b2_7x1_2_relu, n.inception_b2_1x7_2, n.inception_b2_1x7_2_bn, n.inception_b2_1x7_2_scale, \
        n.inception_b2_1x7_2_relu, n.inception_b2_7x1_3, n.inception_b2_7x1_3_bn, n.inception_b2_7x1_3_scale, n.inception_b2_7x1_3_relu, \
        n.inception_b2_1x7_3, n.inception_b2_1x7_3_bn, n.inception_b2_1x7_3_scale, n.inception_b2_1x7_3_relu, n.inception_b2_concat = \
            inception_v3_b(n.inception_b1_concat, outs=160)  # 768x17x17
        n.inception_b3_pool_ave, n.inception_b3_1x1, n.inception_b3_1x1_bn, n.inception_b3_1x1_scale, n.inception_b3_1x1_relu, \
        n.inception_b3_1x1_2, n.inception_b3_1x1_2_bn, n.inception_b3_1x1_2_scale, n.inception_b3_1x1_2_relu, \
        n.inception_b3_1x7_reduce, n.inception_b3_1x7_reduce_bn, n.inception_b3_1x7_reduce_scale, n.inception_b3_1x7_reduce_relu, \
        n.inception_b3_1x7, n.inception_b3_1x7_bn, n.inception_b3_1x7_scale, n.inception_b3_1x7_relu, n.inception_b3_7x1, \
        n.inception_b3_7x1_bn, n.inception_b3_7x1_scale, n.inception_b3_7x1_relu, n.inception_b3_7x1_reduce, n.inception_b3_7x1_reduce_bn, \
        n.inception_b3_7x1_reduce_scale, n.inception_b3_7x1_reduce_relu, n.inception_b3_7x1_2, n.inception_b3_7x1_2_bn, \
        n.inception_b3_7x1_2_scale, n.inception_b3_7x1_2_relu, n.inception_b3_1x7_2, n.inception_b3_1x7_2_bn, n.inception_b3_1x7_2_scale, \
        n.inception_b3_1x7_2_relu, n.inception_b3_7x1_3, n.inception_b3_7x1_3_bn, n.inception_b3_7x1_3_scale, n.inception_b3_7x1_3_relu, \
        n.inception_b3_1x7_3, n.inception_b3_1x7_3_bn, n.inception_b3_1x7_3_scale, n.inception_b3_1x7_3_relu, n.inception_b3_concat = \
            inception_v3_b(n.inception_b2_concat, outs=160)  # 768x17x17
        n.inception_b4_pool_ave, n.inception_b4_1x1, n.inception_b4_1x1_bn, n.inception_b4_1x1_scale, n.inception_b4_1x1_relu, \
        n.inception_b4_1x1_2, n.inception_b4_1x1_2_bn, n.inception_b4_1x1_2_scale, n.inception_b4_1x1_2_relu, \
        n.inception_b4_1x7_reduce, n.inception_b4_1x7_reduce_bn, n.inception_b4_1x7_reduce_scale, n.inception_b4_1x7_reduce_relu, \
        n.inception_b4_1x7, n.inception_b4_1x7_bn, n.inception_b4_1x7_scale, n.inception_b4_1x7_relu, n.inception_b4_7x1, \
        n.inception_b4_7x1_bn, n.inception_b4_7x1_scale, n.inception_b4_7x1_relu, n.inception_b4_7x1_reduce, n.inception_b4_7x1_reduce_bn, \
        n.inception_b4_7x1_reduce_scale, n.inception_b4_7x1_reduce_relu, n.inception_b4_7x1_2, n.inception_b4_7x1_2_bn, \
        n.inception_b4_7x1_2_scale, n.inception_b4_7x1_2_relu, n.inception_b4_1x7_2, n.inception_b4_1x7_2_bn, n.inception_b4_1x7_2_scale, \
        n.inception_b4_1x7_2_relu, n.inception_b4_7x1_3, n.inception_b4_7x1_3_bn, n.inception_b4_7x1_3_scale, n.inception_b4_7x1_3_relu, \
        n.inception_b4_1x7_3, n.inception_b4_1x7_3_bn, n.inception_b4_1x7_3_scale, n.inception_b4_1x7_3_relu, n.inception_b4_concat = \
            inception_v3_b(n.inception_b3_concat, outs=192)  # 768x17x17

        # loss 1
        n.auxiliary_loss_ave_pool = L.Pooling(n.inception_b4_concat, kernel_size=5, stride=3,
                                              pool=P.Pooling.AVE)  # 768x5x5
        n.auxiliary_loss_conv, n.auxiliary_loss_conv_bn, n.auxiliary_loss_conv_scale, n.auxiliary_loss_relu_conv = \
            factorization_conv_bn_scale_relu(n.auxiliary_loss_ave_pool, num_output=128, kernel_size=1)  # 128x1x1
        n.auxiliary_loss_fc = L.InnerProduct(n.auxiliary_loss_conv, num_output=768,
                                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                             weight_filler=dict(type='xavier', std=0.01),
                                             bias_filler=dict(type='constant', value=0))
        n.auxiliary_loss_fc_relu = L.ReLU(n.auxiliary_loss_fc, in_place=True)
        n.auxiliary_loss_classifier = L.InnerProduct(n.auxiliary_loss_fc, num_output=self.classifier_num,
                                                     param=[dict(lr_mult=1, decay_mult=1),
                                                            dict(lr_mult=2, decay_mult=0)],
                                                     weight_filler=dict(type='xavier'),
                                                     bias_filler=dict(type='constant', value=0))
        n.auxiliary_loss = L.SoftmaxWithLoss(n.auxiliary_loss_classifier, n.label, loss_weight=0.4)

        # reduction_v3_b
        n.reduction_b_pool, n.reduction_b_3x3_reduce, n.reduction_b_3x3_reduce_bn, n.reduction_b_3x3_reduce_scale, \
        n.reduction_b_3x3_reduce_relu, n.reduction_b_3x3, n.reduction_b_3x3_bn, n.reduction_b_3x3_scale, n.reduction_b_3x3_relu, \
        n.reduction_b_1x7_reduce, n.reduction_b_1x7_reduce_bn, n.reduction_b_1x7_reduce_scale, n.reduction_b_1x7_reduce_relu, \
        n.reduction_b_1x7, n.reduction_b_1x7_bn, n.reduction_b_1x7_scale, n.reduction_b_1x7_relu, n.reduction_b_7x1, \
        n.reduction_b_7x1_bn, n.reduction_b_7x1_scale, n.reduction_b_7x1_relu, n.reduction_b_3x3_2, n.reduction_b_3x3_2_bn, \
        n.reduction_b_3x3_2_scale, n.reduction_b_3x3_2_relu, n.reduction_b_concat = \
            reduction_v3_b(n.inception_b4_concat)  # 1280x8x8

        #  2 x inception_v3_c
        n.inception_c1_pool, n.inception_c1_1x1, n.inception_c1_1x1_bn, n.inception_c1_1x1_scale, n.inception_c1_1x1_relu, \
        n.inception_c1_1x1_2, n.inception_c1_1x1_2_bn, n.inception_c1_1x1_2_scale, n.inception_c1_1x1_2_relu, \
        n.inception_c1_1x3_reduce, n.inception_c1_1x3_reduce_bn, n.inception_c1_1x3_reduce_scale, n.inception_c1_1x3_reduce_relu, \
        n.inception_c1_1x3, n.inception_c1_1x3_bn, n.inception_c1_1x3_scale, n.inception_c1_1x3_relu, n.inception_c1_3x1, \
        n.inception_c1_3x1_bn, n.inception_c1_3x1_scale, n.inception_c1_3x1_relu, n.inception_c1_3x3_reduce, \
        n.inception_c1_3x3_reduce_bn, n.inception_c1_3x3_reduce_scale, n.inception_c1_3x3_reduce_relu, n.inception_c1_3x3, \
        n.inception_c1_3x3_bn, n.inception_c1_3x3_scale, n.inception_c1_3x3_relu, n.inception_c1_1x3_2, n.inception_c1_1x3_2_bn, \
        n.inception_c1_1x3_2_scale, n.inception_c1_1x3_2_relu, n.inception_c1_3x1_2, n.inception_c1_3x1_2_bn, n.inception_c1_3x1_2_scale, \
        n.inception_c1_3x1_2_relu, n.inception_c1_concat = \
            inception_v3_c(n.reduction_b_concat)  # 2048x8x8
        n.inception_c2_pool, n.inception_c2_1x1, n.inception_c2_1x1_bn, n.inception_c2_1x1_scale, n.inception_c2_1x1_relu, \
        n.inception_c2_1x1_2, n.inception_c2_1x1_2_bn, n.inception_c2_1x1_2_scale, n.inception_c2_1x1_2_relu, \
        n.inception_c2_1x3_reduce, n.inception_c2_1x3_reduce_bn, n.inception_c2_1x3_reduce_scale, n.inception_c2_1x3_reduce_relu, \
        n.inception_c2_1x3, n.inception_c2_1x3_bn, n.inception_c2_1x3_scale, n.inception_c2_1x3_relu, n.inception_c2_3x1, \
        n.inception_c2_3x1_bn, n.inception_c2_3x1_scale, n.inception_c2_3x1_relu, n.inception_c2_3x3_reduce, \
        n.inception_c2_3x3_reduce_bn, n.inception_c2_3x3_reduce_scale, n.inception_c2_3x3_reduce_relu, n.inception_c2_3x3, \
        n.inception_c2_3x3_bn, n.inception_c2_3x3_scale, n.inception_c2_3x3_relu, n.inception_c2_1x3_2, n.inception_c2_1x3_2_bn, \
        n.inception_c2_1x3_2_scale, n.inception_c2_1x3_2_relu, n.inception_c2_3x1_2, n.inception_c2_3x1_2_bn, n.inception_c2_3x1_2_scale, \
        n.inception_c2_3x1_2_relu, n.inception_c2_concat = \
            inception_v3_c(n.inception_c1_concat, pool=P.Pooling.MAX)  # 2048x8x8

        # loss 2
        n.pool_8x8_s1 = L.Pooling(n.inception_c2_concat, kernel_size=8, pool=P.Pooling.AVE)
        n.pool_8x8_s1_drop = L.Dropout(n.pool_8x8_s1, dropout_param=dict(dropout_ratio=0.2))
        n.classifier = L.InnerProduct(n.pool_8x8_s1_drop, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
                                         accuracy_param=dict(top_k=5))

        return n.to_proto()
