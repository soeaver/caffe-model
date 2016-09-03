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
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def factorization_conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


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


def eltwise_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def stem_resnet_v2_299x299(bottom):
    """
    input:3x299x299
    output:320x35x35
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

    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(pool2_3x3_s2, num_output=96, kernel_size=1)  # 96x35x35

    conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_scale, conv_5x5_reduce_relu = \
        factorization_conv_bn_scale_relu(pool2_3x3_s2, num_output=48, kernel_size=1)  # 48x35x35
    conv_5x5, conv_5x5_bn, conv_5x5_scale, conv_5x5_relu = \
        factorization_conv_bn_scale_relu(conv_5x5_reduce, num_output=64, kernel_size=5, pad=2)  # 64x35x35

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(pool2_3x3_s2, num_output=64, kernel_size=1)  # 64x35x35
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=96, kernel_size=3, pad=1)  # 96x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3, num_output=96, kernel_size=3, pad=1)  # 96x35x35

    ave_pool = L.Pooling(pool2_3x3_s2, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 192x35x35
    conv_1x1_ave, conv_1x1_ave_bn, conv_1x1_ave_scale, conv_1x1_ave_relu = \
        factorization_conv_bn_scale_relu(ave_pool, num_output=64, kernel_size=1)  # 64x35x35

    concat = L.Concat(conv_1x1, conv_5x5, conv_3x3_2, conv_1x1_ave)  # 320(96+64+96+64)x35x35

    return conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_relu, conv2_3x3_s1, conv2_3x3_s1_bn, \
           conv2_3x3_s1_scale, conv2_3x3_s1_relu, conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu, \
           pool1_3x3_s2, conv4_3x3_reduce, conv4_3x3_reduce_bn, conv4_3x3_reduce_scale, conv4_3x3_reduce_relu, \
           conv4_3x3, conv4_3x3_bn, conv4_3x3_scale, conv4_3x3_relu, pool2_3x3_s2, conv_1x1, conv_1x1_bn, conv_1x1_scale, \
           conv_1x1_relu, conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_scale, conv_5x5_reduce_relu, \
           conv_5x5, conv_5x5_bn, conv_5x5_scale, conv_5x5_relu, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, \
           conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2, conv_3x3_2_bn, \
           conv_3x3_2_scale, conv_3x3_2_relu, ave_pool, conv_1x1_ave, conv_1x1_ave_bn, conv_1x1_ave_scale, conv_1x1_ave_relu, \
           concat


def inception_resnet_v2_a(bottom):
    """
    input:320x35x35
    output:320x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=32, kernel_size=3, pad=1)  # 32x35x35

    conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2_reduce, num_output=48, kernel_size=3, pad=1)  # 48x35x35
    conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2, num_output=64, kernel_size=3, pad=1)  # 64x35x35

    concat = L.Concat(conv_1x1, conv_3x3, conv_3x3_3)  # 128(32+32+64)x35x35
    conv_up, conv_up_bn, conv_up_scale = \
        factorization_conv_bn_scale(concat, num_output=320, kernel_size=1)  # 320x35x35

    residual_eltwise, residual_eltwise_relu = eltwise_relu(bottom, conv_up)  # 320x35x35

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_3x3_reduce, conv_3x3_reduce_bn, \
           conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, \
           conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, \
           conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, \
           conv_3x3_3_relu, concat, conv_up, conv_up_bn, conv_up_scale, residual_eltwise, residual_eltwise_relu


def reduction_resnet_v2_a(bottom):
    """
    input:320x35x35
    output:1088x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=384, kernel_size=3, stride=2)  # 384x17x17

    conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2_reduce, num_output=256, kernel_size=3, stride=1, pad=1)  # 256x35x35
    conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2, num_output=384, kernel_size=3, stride=2)  # 384x17x17

    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 320x17x17

    concat = L.Concat(conv_3x3, conv_3x3_3, pool)  # 1088(320+384+384)x17x17 

    return conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, pool, concat


def inception_resnet_v2_b(bottom):
    """
    input:1088x17x17
    output:1088x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x17x17

    conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_scale, conv_1x7_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=128, kernel_size=1)  # 128x17x17
    conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu = \
        factorization_conv_mxn(conv_1x7_reduce, num_output=160, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 160x17x17
    conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu = \
        factorization_conv_mxn(conv_1x7, num_output=192, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 192x17x17

    concat = L.Concat(conv_1x1, conv_7x1)  # 384(192+192)x17x17
    conv_up, conv_up_bn, conv_up_scale = \
        factorization_conv_bn_scale(concat, num_output=1088, kernel_size=1)  # 1088x17x17 

    residual_eltwise, residual_eltwise_relu = eltwise_relu(bottom, conv_up)  # 1088x17x17

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x7_reduce, conv_1x7_reduce_bn, \
           conv_1x7_reduce_scale, conv_1x7_reduce_relu, conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu, \
           conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu, concat, conv_up, conv_up_bn, conv_up_scale, \
           residual_eltwise, residual_eltwise_relu


def reduction_resnet_v2_b(bottom):
    """
    input:1088x17x17
    output:2080x8x8
    :param bottom: bottom layer
    :return: layers
    """
    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=384, kernel_size=3, stride=2)  # 384x8x8

    conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2_reduce, num_output=288, kernel_size=3, stride=2)  # 288x8x8

    conv_3x3_3_reduce, conv_3x3_3_reduce_bn, conv_3x3_3_reduce_scale, conv_3x3_3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_3_reduce, num_output=288, kernel_size=3, pad=1)  # 288x17x17
    conv_3x3_4, conv_3x3_4_bn, conv_3x3_4_scale, conv_3x3_4_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_3, num_output=320, kernel_size=3, stride=2)  # 320x8x8

    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 1088x8x8

    concat = L.Concat(conv_3x3, conv_3x3_2, conv_3x3_4, pool)  # 2080(1088+384+288+320)x8x8 

    return conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, \
           conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3_reduce, conv_3x3_3_reduce_bn, conv_3x3_3_reduce_scale, conv_3x3_3_reduce_relu, \
           conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, conv_3x3_4, conv_3x3_4_bn, conv_3x3_4_scale, \
           conv_3x3_4_relu, pool, concat


def inception_resnet_v2_c(bottom):
    """
    input:2080x8x8
    output:2080x8x8
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x8x8

    conv_1x3_reduce, conv_1x3_reduce_bn, conv_1x3_reduce_scale, conv_1x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x8x8
    conv_1x3, conv_1x3_bn, conv_1x3_scale, conv_1x3_relu = \
        factorization_conv_mxn(conv_1x3_reduce, num_output=224, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 224x8x8
    conv_3x1, conv_3x1_bn, conv_3x1_scale, conv_3x1_relu = \
        factorization_conv_mxn(conv_1x3, num_output=256, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 256x8x8

    concat = L.Concat(conv_1x1, conv_3x1)  # 448(192+256)x8x8
    conv_up, conv_up_bn, conv_up_scale = \
        factorization_conv_bn_scale(concat, num_output=2080, kernel_size=1)  # 2080x8x8 

    residual_eltwise, residual_eltwise_relu = eltwise_relu(bottom, conv_up)  # 2080x8x8  

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x3_reduce, conv_1x3_reduce_bn, \
           conv_1x3_reduce_scale, conv_1x3_reduce_relu, conv_1x3, conv_1x3_bn, conv_1x3_scale, conv_1x3_relu, \
           conv_3x1, conv_3x1_bn, conv_3x1_scale, conv_3x1_relu, concat, conv_up, conv_up_bn, conv_up_scale, \
           residual_eltwise, residual_eltwise_relu


string_a = 'n.inception_resnet_v2_a(order)_1x1, n.inception_resnet_v2_a(order)_1x1_bn, n.inception_resnet_v2_a(order)_1x1_scale, \
        n.inception_resnet_v2_a(order)_1x1_relu, n.inception_resnet_v2_a(order)_3x3_reduce, n.inception_resnet_v2_a(order)_3x3_reduce_bn, \
        n.inception_resnet_v2_a(order)_3x3_reduce_scale, n.inception_resnet_v2_a(order)_3x3_reduce_relu, n.inception_resnet_v2_a(order)_3x3, \
        n.inception_resnet_v2_a(order)_3x3_bn, n.inception_resnet_v2_a(order)_3x3_scale, n.inception_resnet_v2_a(order)_3x3_relu, \
        n.inception_resnet_v2_a(order)_3x3_2_reduce, n.inception_resnet_v2_a(order)_3x3_2_reduce_bn, n.inception_resnet_v2_a(order)_3x3_2_reduce_scale, \
        n.inception_resnet_v2_a(order)_3x3_2_reduce_relu, n.inception_resnet_v2_a(order)_3x3_2, n.inception_resnet_v2_a(order)_3x3_2_bn, \
        n.inception_resnet_v2_a(order)_3x3_2_scale, n.inception_resnet_v2_a(order)_3x3_2_relu, n.inception_resnet_v2_a(order)_3x3_3, \
        n.inception_resnet_v2_a(order)_3x3_3_bn, n.inception_resnet_v2_a(order)_3x3_3_scale, n.inception_resnet_v2_a(order)_3x3_3_relu, \
        n.inception_resnet_v2_a(order)_concat, n.inception_resnet_v2_a(order)_up, n.inception_resnet_v2_a(order)_up_bn, \
        n.inception_resnet_v2_a(order)_up_scale, n.inception_resnet_v2_a(order)_residual_eltwise, \
        n.inception_resnet_v2_a(order)_residual_eltwise_relu = \
            inception_resnet_v2_a(bottom)'

string_b = 'n.inception_resnet_v2_b(order)_1x1, n.inception_resnet_v2_b(order)_1x1_bn, n.inception_resnet_v2_b(order)_1x1_scale, \
        n.inception_resnet_v2_b(order)_1x1_relu, n.inception_resnet_v2_b(order)_1x7_reduce, n.inception_resnet_v2_b(order)_1x7_reduce_bn, \
        n.inception_resnet_v2_b(order)_1x7_reduce_scale, n.inception_resnet_v2_b(order)_1x7_reduce_relu, n.inception_resnet_v2_b(order)_1x7, \
        n.inception_resnet_v2_b(order)_1x7_bn, n.inception_resnet_v2_b(order)_1x7_scale, n.inception_resnet_v2_b(order)_1x7_relu, \
        n.inception_resnet_v2_b(order)_7x1, n.inception_resnet_v2_b(order)_7x1_bn, n.inception_resnet_v2_b(order)_7x1_scale, \
        n.inception_resnet_v2_b(order)_7x1_relu, n.inception_resnet_v2_b(order)_concat, n.inception_resnet_v2_b(order)_up, \
        n.inception_resnet_v2_b(order)_up_bn, n.inception_resnet_v2_b(order)_up_scale, n.inception_resnet_v2_b(order)_residual_eltwise, \
        n.inception_resnet_v2_b(order)_residual_eltwise_relu \
            = inception_resnet_v2_b(bottom)'

string_c = 'n.inception_resnet_v2_c(order)_1x1, n.inception_resnet_v2_c(order)_1x1_bn, n.inception_resnet_v2_c(order)_1x1_scale, \
        n.inception_resnet_v2_c(order)_1x1_relu, n.inception_resnet_v2_c(order)_1x3_reduce, n.inception_resnet_v2_c(order)_1x3_reduce_bn, \
        n.inception_resnet_v2_c(order)_1x3_reduce_scale, n.inception_resnet_v2_c(order)_1x3_reduce_relu, n.inception_resnet_v2_c(order)_1x3, \
        n.inception_resnet_v2_c(order)_1x3_bn, n.inception_resnet_v2_c(order)_1x3_scale, n.inception_resnet_v2_c(order)_1x3_relu, \
        n.inception_resnet_v2_c(order)_3x1, n.inception_resnet_v2_c(order)_3x1_bn, n.inception_resnet_v2_c(order)_3x1_scale, \
        n.inception_resnet_v2_c(order)_3x1_relu, n.inception_resnet_v2_c(order)_concat, n.inception_resnet_v2_c(order)_up, \
        n.inception_resnet_v2_c(order)_up_bn, n.inception_resnet_v2_c(order)_up_scale, n.inception_resnet_v2_c(order)_residual_eltwise, \
        n.inception_resnet_v2_c(order)_residual_eltwise_relu = \
            inception_resnet_v2_c(bottom)'


class InceptionResNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def inception_resnet_v2_proto(self, batch_size, phase='TRAIN'):
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
        n.pool1_3x3_s2, n.conv4_3x3_reduce, n.conv4_3x3_reduce_bn, n.conv4_3x3_reduce_scale, n.conv4_3x3_reduce_relu, \
        n.conv4_3x3, n.conv4_3x3_bn, n.conv4_3x3_scale, n.conv4_relu_3x3, n.pool2_3x3_s2, n.conv5_1x1, n.conv5_1x1_bn, n.conv5_1x1_scale, \
        n.conv5_1x1_relu, n.conv5_5x5_reduce, n.conv5_5x5_reduce_bn, n.conv5_5x5_reduce_scale, n.conv5_5x5_reduce_relu, \
        n.conv5_5x5, n.conv5_5x5_bn, n.conv5_5x5_scale, n.conv5_5x5_relu, n.conv5_3x3_reduce, n.conv5_3x3_reduce_bn, n.conv5_3x3_reduce_scale, \
        n.conv5_3x3_reduce_relu, n.conv5_3x3, n.conv5_3x3_bn, n.conv5_3x3_scale, n.conv5_3x3_relu, n.conv5_3x3_2, n.conv5_3x3_2_bn, \
        n.conv5_3x3_2_scale, n.conv5_3x3_2_relu, n.ave_pool, n.conv5_1x1_ave, n.conv5_1x1_ave_bn, n.conv5_1x1_ave_scale, n.conv5_1x1_ave_relu, \
        n.stem_concat = stem_resnet_v2_299x299(n.data)  # 320x35x35

        # 10 x inception_resnet_v2_a
        for i in xrange(10):
            if i == 0:
                bottom = 'n.stem_concat'
            else:
                bottom = 'n.inception_resnet_v2_a(order)_residual_eltwise'.replace('(order)', str(i))
            exec (string_a.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 384x35x35

        # reduction_resnet_v2_a
        n.reduction_a_3x3, n.reduction_a_3x3_bn, n.reduction_a_3x3_scale, n.reduction_a_3x3_relu, \
        n.reduction_a_3x3_2_reduce, n.reduction_a_3x3_2_reduce_bn, n.reduction_a_3x3_2_reduce_scale, \
        n.reduction_a_3x3_2_reduce_relu, n.reduction_a_3x3_2, n.reduction_a_3x3_2_bn, n.reduction_a_3x3_2_scale, \
        n.reduction_a_3x3_2_relu, n.reduction_a_3x3_3, n.reduction_a_3x3_3_bn, n.reduction_a_3x3_3_scale, \
        n.reduction_a_3x3_3_relu, n.reduction_a_pool, n.reduction_a_concat = \
            reduction_resnet_v2_a(n.inception_resnet_v2_a10_residual_eltwise)  # 1088x17x17

        # 20 x inception_resnet_v2_b
        for i in xrange(20):
            if i == 0:
                bottom = 'n.reduction_a_concat'
            else:
                bottom = 'n.inception_resnet_v2_b(order)_residual_eltwise'.replace('(order)', str(i))
            exec (string_b.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 1088x17x17

        # reduction_resnet_v2_b
        n.reduction_b_3x3_reduce, n.reduction_b_3x3_reduce_bn, n.reduction_b_3x3_reduce_scale, \
        n.reduction_b_3x3_reduce_relu, n.reduction_b_3x3, n.reduction_b_3x3_bn, n.reduction_b_3x3_scale, \
        n.reduction_b_3x3_relu, n.reduction_b_3x3_2_reduce, n.reduction_b_3x3_2_reduce_bn, n.reduction_b_3x3_2_reduce_scale, \
        n.reduction_b_3x3_2_reduce_relu, n.reduction_b_3x3_2, n.reduction_b_3x3_2_bn, n.reduction_b_3x3_2_scale, \
        n.reduction_b_3x3_2_relu, n.reduction_b_3x3_3_reduce, n.reduction_b_3x3_3_reduce_bn, n.reduction_b_3x3_3_reduce_scale, \
        n.reduction_b_3x3_3_reduce_relu, n.reduction_b_3x3_3, n.reduction_b_3x3_3_bn, n.reduction_b_3x3_3_scale, \
        n.reduction_b_3x3_3_relu, n.reduction_b_3x3_4, n.reduction_b_3x3_4_bn, n.reduction_b_3x3_4_scale, \
        n.reduction_b_3x3_4_relu, n.reduction_b_pool, n.reduction_b_concat = \
            reduction_resnet_v2_b(n.inception_resnet_v2_b20_residual_eltwise)  # 2080x8x8

        # 9 x inception_resnet_v2_c
        for i in xrange(9):
            if i == 0:
                bottom = 'n.reduction_b_concat'
            else:
                bottom = 'n.inception_resnet_v2_c(order)_residual_eltwise'.replace('(order)', str(i))
            exec (string_c.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 2080x8x8

        n.inception_resnet_v2_c10_1x1, n.inception_resnet_v2_c10_1x1_bn, n.inception_resnet_v2_c10_1x1_scale, \
        n.inception_resnet_v2_c10_1x1_relu = \
            factorization_conv_bn_scale_relu(n.inception_resnet_v2_c9_residual_eltwise, num_output=192,
                                             kernel_size=1)  # 192x8x8

        n.inception_resnet_v2_c10_1x3_reduce, n.inception_resnet_v2_c10_1x3_reduce_bn, \
        n.inception_resnet_v2_c10_1x3_reduce_scale, n.inception_resnet_v2_c10_1x3_reduce_relu = \
            factorization_conv_bn_scale_relu(n.inception_resnet_v2_c9_residual_eltwise, num_output=192,
                                             kernel_size=1)  # 192x8x8
        n.inception_resnet_v2_c10_1x3, n.inception_resnet_v2_c10_1x3_bn, n.inception_resnet_v2_c10_1x3_scale, \
        n.inception_resnet_v2_c10_1x3_relu = \
            factorization_conv_mxn(n.inception_resnet_v2_c10_1x3_reduce, num_output=224, kernel_h=1, kernel_w=3,
                                   pad_h=0, pad_w=1)  # 224x8x8
        n.inception_resnet_v2_c10_3x1, n.inception_resnet_v2_c10_3x1_bn, n.inception_resnet_v2_c10_3x1_scale, \
        n.inception_resnet_v2_c10_3x1_relu = \
            factorization_conv_mxn(n.inception_resnet_v2_c10_1x3, num_output=256, kernel_h=3, kernel_w=1, pad_h=1,
                                   pad_w=0)  # 256x8x8

        n.inception_resnet_v2_c10_concat = L.Concat(n.inception_resnet_v2_c10_1x1,
                                                    n.inception_resnet_v2_c10_3x1)  # 448(192+256)x8x8
        n.inception_resnet_v2_c10_up, n.inception_resnet_v2_c10_up_bn, n.inception_resnet_v2_c10_up_scale = \
            factorization_conv_bn_scale(n.inception_resnet_v2_c10_concat, num_output=2080,
                                        kernel_size=1)  # 2080x8x8

        n.inception_resnet_v2_c10_residual_eltwise = \
            L.Eltwise(n.inception_resnet_v2_c9_residual_eltwise, n.inception_resnet_v2_c10_up,
                      eltwise_param=dict(operation=1))  # 2080x8x8

        n.conv6_1x1, n.conv6_1x1_bn, n.conv6_1x1_scale, n.conv6_1x1_relu = \
            factorization_conv_bn_scale_relu(n.inception_resnet_v2_c10_residual_eltwise, num_output=1536,
                                             kernel_size=1)  # 1536x8x8

        n.pool_8x8_s1 = L.Pooling(n.conv6_1x1,
                                  pool=P.Pooling.AVE,
                                  global_pooling=True)  # 1536x1x1
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
