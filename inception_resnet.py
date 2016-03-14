import caffe
from caffe import layers as L
from caffe import params as P


def stem_299x299(bottom):
    """
    input:3x299x299
    output:384x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2 = L.Convolution(bottom, kernel_size=3, num_output=32, stride=2,
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 weight_filler=dict(type='xavier', std=0.01),
                                 bias_filler=dict(type='constant', value=0.2))  # 32x149x149
    conv1_3x3_s2_bn = L.BatchNorm(conv1_3x3_s2, use_global_stats=False, in_place=True)
    conv1_3x3_s2_scale = L.Scale(conv1_3x3_s2, scale_param=dict(bias_term=True), in_place=True)
    conv1_3x3_s2_relu = L.ReLU(conv1_3x3_s2, in_place=True)

    conv2_3x3_s1 = L.Convolution(conv1_3x3_s2, kernel_size=3, num_output=32, stride=1,
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 weight_filler=dict(type='xavier', std=0.01),
                                 bias_filler=dict(type='constant', value=0.2))  # 32x147x147
    conv2_3x3_s1_bn = L.BatchNorm(conv2_3x3_s1, use_global_stats=False, in_place=True)
    conv2_3x3_s1_scale = L.Scale(conv2_3x3_s1, scale_param=dict(bias_term=True), in_place=True)
    conv2_3x3_s1_relu = L.ReLU(conv2_3x3_s1, in_place=True)

    conv3_3x3_s1 = L.Convolution(conv2_3x3_s1, kernel_size=3, num_output=64, stride=1, pad=1,
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 weight_filler=dict(type='xavier', std=0.01),
                                 bias_filler=dict(type='constant', value=0.2))  # 64x147x147
    conv3_3x3_s1_bn = L.BatchNorm(conv3_3x3_s1, use_global_stats=False, in_place=True)
    conv3_3x3_s1_scale = L.Scale(conv3_3x3_s1, scale_param=dict(bias_term=True), in_place=True)
    conv3_3x3_s1_relu = L.ReLU(conv3_3x3_s1, in_place=True)

    inception_stem1_3x3_s2 = L.Convolution(conv3_3x3_s1, kernel_size=3, num_output=96, stride=2,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           weight_filler=dict(type='xavier', std=0.01),
                                           bias_filler=dict(type='constant', value=0.2))  # 96x73x73
    inception_stem1_3x3_s2_bn = L.BatchNorm(inception_stem1_3x3_s2, use_global_stats=False, in_place=True)
    inception_stem1_3x3_s2_scale = L.Scale(inception_stem1_3x3_s2, scale_param=dict(bias_term=True),
                                           in_place=True)
    inception_stem1_3x3_s2_relu = L.ReLU(inception_stem1_3x3_s2, in_place=True)
    inception_stem1_pool = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2,
                                     pool=P.Pooling.MAX)  # 64x73x73
    inception_stem1 = L.Concat(inception_stem1_3x3_s2, inception_stem1_pool)  # 160x73x73

    inception_stem2_3x3_reduce = L.Convolution(inception_stem1, kernel_size=1, num_output=64,
                                               param=[dict(lr_mult=1, decay_mult=1),
                                                      dict(lr_mult=2, decay_mult=0)],
                                               weight_filler=dict(type='xavier', std=0.01),
                                               bias_filler=dict(type='constant', value=0.2))  # 64x73x73
    inception_stem2_3x3_reduce_bn = L.BatchNorm(inception_stem2_3x3_reduce, use_global_stats=False,
                                                in_place=True)
    inception_stem2_3x3_reduce_scale = L.Scale(inception_stem2_3x3_reduce,
                                               scale_param=dict(bias_term=True), in_place=True)
    inception_stem2_3x3_reduce_relu = L.ReLU(inception_stem2_3x3_reduce, in_place=True)
    inception_stem2_3x3 = L.Convolution(inception_stem2_3x3_reduce, kernel_size=3, num_output=96,
                                        param=[dict(lr_mult=1, decay_mult=1),
                                               dict(lr_mult=2, decay_mult=0)],
                                        weight_filler=dict(type='xavier', std=0.01),
                                        bias_filler=dict(type='constant', value=0.2))  # 96x71x71
    inception_stem2_3x3_bn = L.BatchNorm(inception_stem2_3x3, use_global_stats=False, in_place=True)
    inception_stem2_3x3_scale = L.Scale(inception_stem2_3x3, scale_param=dict(bias_term=True), in_place=True)
    inception_stem2_3x3_relu = L.ReLU(inception_stem2_3x3, in_place=True)

    inception_stem2_7x1_reduce = L.Convolution(inception_stem1, kernel_size=1, num_output=64,
                                               param=[dict(lr_mult=1, decay_mult=1),
                                                      dict(lr_mult=2, decay_mult=0)],
                                               weight_filler=dict(type='xavier', std=0.01),
                                               bias_filler=dict(type='constant', value=0.2))  # 64x73x73
    inception_stem2_7x1_reduce_bn = L.BatchNorm(inception_stem2_7x1_reduce, use_global_stats=False,
                                                in_place=True)
    inception_stem2_7x1_reduce_scale = L.Scale(inception_stem2_7x1_reduce,
                                               scale_param=dict(bias_term=True), in_place=True)
    inception_stem2_7x1_reduce_relu = L.ReLU(inception_stem2_7x1_reduce, in_place=True)
    inception_stem2_7x1 = L.Convolution(inception_stem2_7x1_reduce, kernel_h=7, kernel_w=1, num_output=64,
                                        pad_h=3, pad_w=0, stride=1,
                                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                        weight_filler=dict(type='xavier'),
                                        bias_filler=dict(type='constant', value=0))  # 64x73x73
    inception_stem2_7x1_bn = L.BatchNorm(inception_stem2_7x1, use_global_stats=False, in_place=True)
    inception_stem2_7x1_scale = L.Scale(inception_stem2_7x1, scale_param=dict(bias_term=True), in_place=True)
    inception_stem2_7x1_relu = L.ReLU(inception_stem2_7x1, in_place=True)
    inception_stem2_1x7 = L.Convolution(inception_stem2_7x1, kernel_h=1, kernel_w=7, num_output=64,
                                        pad_h=0, pad_w=3, stride=1,
                                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                        weight_filler=dict(type='xavier'),
                                        bias_filler=dict(type='constant', value=0))  # 64x73x73
    inception_stem2_1x7_bn = L.BatchNorm(inception_stem2_1x7, use_global_stats=False, in_place=True)
    inception_stem2_1x7_scale = L.Scale(inception_stem2_1x7, scale_param=dict(bias_term=True), in_place=True)
    inception_stem2_1x7_relu = L.ReLU(inception_stem2_1x7, in_place=True)
    inception_stem2_3x3_2 = L.Convolution(inception_stem2_1x7, kernel_size=3, num_output=96,
                                          param=[dict(lr_mult=1, decay_mult=1),
                                                 dict(lr_mult=2, decay_mult=0)],
                                          weight_filler=dict(type='xavier', std=0.01),
                                          bias_filler=dict(type='constant', value=0.2))  # 96x71x71
    inception_stem2_3x3_2_bn = L.BatchNorm(inception_stem2_3x3_2, use_global_stats=False, in_place=True)
    inception_stem2_3x3_2_scale = L.Scale(inception_stem2_3x3_2, scale_param=dict(bias_term=True),
                                          in_place=True)
    inception_stem2_3x3_2_relu = L.ReLU(inception_stem2_3x3_2, in_place=True)
    inception_stem2 = L.Concat(inception_stem2_3x3, inception_stem2_3x3_2)  # 192x71x71

    inception_stem3_3x3_s2 = L.Convolution(inception_stem2, kernel_size=3, num_output=192, stride=2,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           weight_filler=dict(type='xavier', std=0.01),
                                           bias_filler=dict(type='constant', value=0.2))  # 192x35x35
    inception_stem3_3x3_s2_bn = L.BatchNorm(inception_stem3_3x3_s2, use_global_stats=False, in_place=True)
    inception_stem3_3x3_s2_scale = L.Scale(inception_stem3_3x3_s2, scale_param=dict(bias_term=True),
                                           in_place=True)
    inception_stem3_3x3_s2_relu = L.ReLU(inception_stem3_3x3_s2, in_place=True)
    inception_stem3_pool = L.Pooling(inception_stem2, kernel_size=3, stride=2,
                                     pool=P.Pooling.MAX)  # 192x35x35
    inception_stem3 = L.Concat(inception_stem3_3x3_s2, inception_stem3_pool)  # 384x35x35

    return conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_s2_relu, conv2_3x3_s1, conv2_3x3_s1_bn, \
           conv2_3x3_s1_scale, conv2_3x3_s1_relu, conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu, \
           inception_stem1_3x3_s2, inception_stem1_3x3_s2_bn, inception_stem1_3x3_s2_scale, inception_stem1_3x3_s2_relu, \
           inception_stem1_pool, inception_stem1, inception_stem2_3x3_reduce, inception_stem2_3x3_reduce_bn, \
           inception_stem2_3x3_reduce_scale, inception_stem2_3x3_reduce_relu, inception_stem2_3x3, \
           inception_stem2_3x3_bn, inception_stem2_3x3_scale, inception_stem2_3x3_relu, inception_stem2_7x1_reduce, \
           inception_stem2_7x1_reduce_bn, inception_stem2_7x1_reduce_scale, inception_stem2_7x1_reduce_relu, \
           inception_stem2_7x1, inception_stem2_7x1_bn, inception_stem2_7x1_scale, inception_stem2_7x1_relu, \
           inception_stem2_1x7, inception_stem2_1x7_bn, inception_stem2_1x7_scale, inception_stem2_1x7_relu, \
           inception_stem2_3x3_2, inception_stem2_3x3_2_bn, inception_stem2_3x3_2_scale, inception_stem2_3x3_2_relu, \
           inception_stem2, inception_stem3_3x3_s2, inception_stem3_3x3_s2_bn, inception_stem3_3x3_s2_scale, \
           inception_stem3_3x3_s2_relu, inception_stem3_pool, inception_stem3


def factorization_inception_resnet_a(bottom):
    """
    input:384x35x35
    output:384x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=32, stride=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 32x35x35
    conv_1x1_bn = L.BatchNorm(conv_1x1, use_global_stats=False, in_place=True)
    conv_1x1_scale = L.Scale(conv_1x1, scale_param=dict(bias_term=True), in_place=True)
    conv_1x1_relu = L.ReLU(conv_1x1, in_place=True)

    conv_3x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=32, stride=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0))  # 32x35x35
    conv_3x3_reduce_bn = L.BatchNorm(conv_3x3_reduce, use_global_stats=False, in_place=True)
    conv_3x3_reduce_scale = L.Scale(conv_3x3_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_reduce_relu = L.ReLU(conv_3x3_reduce, in_place=True)

    conv_3x3 = L.Convolution(conv_3x3_reduce, kernel_size=3, num_output=32, stride=1, pad=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 32x35x35
    conv_3x3_bn = L.BatchNorm(conv_3x3, use_global_stats=False, in_place=True)
    conv_3x3_scale = L.Scale(conv_3x3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_relu = L.ReLU(conv_3x3, in_place=True)

    conv_3x3_2_reduce = L.Convolution(bottom, kernel_size=1, num_output=32, stride=1,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))  # 32x35x35
    conv_3x3_2_reduce_bn = L.BatchNorm(conv_3x3_2_reduce, use_global_stats=False, in_place=True)
    conv_3x3_2_reduce_scale = L.Scale(conv_3x3_2_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_reduce_relu = L.ReLU(conv_3x3_2_reduce, in_place=True)

    conv_3x3_2 = L.Convolution(conv_3x3_2_reduce, kernel_size=3, num_output=48, stride=1, pad=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 48x35x35
    conv_3x3_2_bn = L.BatchNorm(conv_3x3_2, use_global_stats=False, in_place=True)
    conv_3x3_2_scale = L.Scale(conv_3x3_2, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_relu = L.ReLU(conv_3x3_2, in_place=True)

    conv_3x3_3 = L.Convolution(conv_3x3_2, kernel_size=3, num_output=64, stride=1, pad=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 64x35x35
    conv_3x3_3_bn = L.BatchNorm(conv_3x3_3, use_global_stats=False, in_place=True)
    conv_3x3_3_scale = L.Scale(conv_3x3_3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_3_relu = L.ReLU(conv_3x3_3, in_place=True)

    concat = L.Concat(conv_1x1, conv_3x3, conv_3x3_3)  # 128x35x35

    conv_1x1_2 = L.Convolution(concat, kernel_size=1, num_output=384, stride=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 384x35x35
    conv_1x1_2_bn = L.BatchNorm(conv_1x1_2, use_global_stats=False, in_place=True)
    conv_1x1_2_scale = L.Scale(conv_1x1_2, scale_param=dict(bias_term=True), in_place=True)
    # conv_1x1_2_relu = L.ReLU(conv_1x1_2_scale, in_place=True) # linear activation

    residual_eltwise = L.Eltwise(bottom, conv_1x1_2,
                                 eltwise_param=dict(operation=1))

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_3x3_reduce, conv_3x3_reduce_bn, \
           conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, \
           conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, \
           conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, \
           conv_3x3_3_relu, concat, conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, residual_eltwise


def reduction_a(bottom):
    """
    input:384x35x35
    output:1152x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 384x17x17

    conv_3x3 = L.Convolution(bottom, kernel_size=3, num_output=384, stride=2,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 384x17x17
    conv_3x3_bn = L.BatchNorm(conv_3x3, use_global_stats=False, in_place=True)
    conv_3x3_scale = L.Scale(conv_3x3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_relu = L.ReLU(conv_3x3, in_place=True)

    conv_3x3_2_reduce = L.Convolution(bottom, kernel_size=1, num_output=256, stride=1,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))  # 256x35x35
    conv_3x3_2_reduce_bn = L.BatchNorm(conv_3x3_2_reduce, use_global_stats=False, in_place=True)
    conv_3x3_2_reduce_scale = L.Scale(conv_3x3_2_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_reduce_relu = L.ReLU(conv_3x3_2_reduce, in_place=True)

    conv_3x3_2 = L.Convolution(conv_3x3_2_reduce, kernel_size=3, num_output=256, stride=1, pad=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 256x35x35
    conv_3x3_2_bn = L.BatchNorm(conv_3x3_2, use_global_stats=False, in_place=True)
    conv_3x3_2_scale = L.Scale(conv_3x3_2, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_relu = L.ReLU(conv_3x3_2, in_place=True)

    conv_3x3_3 = L.Convolution(conv_3x3_2, kernel_size=3, num_output=384, stride=2,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 384x17x17
    conv_3x3_3_bn = L.BatchNorm(conv_3x3_3, use_global_stats=False, in_place=True)
    conv_3x3_3_scale = L.Scale(conv_3x3_3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_3_relu = L.ReLU(conv_3x3_3, in_place=True)

    concat = L.Concat(pool, conv_3x3, conv_3x3_3)  # 1152x17x17

    return pool, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, concat


def factorization_inception_resnet_b(bottom):
    """
    input:1152x17x17
    output:1152x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=192, stride=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 192x17x17
    conv_1x1_bn = L.BatchNorm(conv_1x1, use_global_stats=False, in_place=True)
    conv_1x1_scale = L.Scale(conv_1x1, scale_param=dict(bias_term=True), in_place=True)
    conv_1x1_relu = L.ReLU(conv_1x1, in_place=True)

    conv_1x7_reduce = L.Convolution(bottom, kernel_size=1, num_output=128, stride=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0))  # 128x17x17
    conv_1x7_reduce_bn = L.BatchNorm(conv_1x7_reduce, use_global_stats=False, in_place=True)
    conv_1x7_reduce_scale = L.Scale(conv_1x7_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_1x7_reduce_relu = L.ReLU(conv_1x7_reduce, in_place=True)

    conv_1x7 = L.Convolution(conv_1x7_reduce, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3, stride=1, num_output=160,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 160x17x17
    conv_1x7_bn = L.BatchNorm(conv_1x7, use_global_stats=False, in_place=True)
    conv_1x7_scale = L.Scale(conv_1x7, scale_param=dict(bias_term=True), in_place=True)
    conv_1x7_relu = L.ReLU(conv_1x7, in_place=True)

    conv_7x1 = L.Convolution(conv_1x7, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0, stride=1, num_output=192,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 192x17x17
    conv_7x1_bn = L.BatchNorm(conv_7x1, use_global_stats=False, in_place=True)
    conv_7x1_scale = L.Scale(conv_7x1, scale_param=dict(bias_term=True), in_place=True)
    conv_7x1_relu = L.ReLU(conv_7x1, in_place=True)

    concat = L.Concat(conv_1x1, conv_7x1)  # 384x17x17

    conv_1x1_2 = L.Convolution(concat, kernel_size=1, num_output=1152, stride=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 1152x17x17
    conv_1x1_2_bn = L.BatchNorm(conv_1x1_2, use_global_stats=False, in_place=True)
    conv_1x1_2_scale = L.Scale(conv_1x1_2, scale_param=dict(bias_term=True), in_place=True)
    # conv_1x1_2_relu = L.ReLU(conv_1x1_2_scale, in_place=True)  # linear activation

    residual_eltwise = L.Eltwise(bottom, conv_1x1_2,
                                 eltwise_param=dict(operation=1))

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x7_reduce, conv_1x7_reduce_bn, \
           conv_1x7_reduce_scale, conv_1x7_reduce_relu, conv_1x7, conv_1x7_bn, conv_1x7_scale, conv_1x7_relu, \
           conv_7x1, conv_7x1_bn, conv_7x1_scale, conv_7x1_relu, concat, conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, \
           residual_eltwise


def reduction_b(bottom):
    """
    input:1152x17x17
    output:2048x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 1152x8x8

    conv_3x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=256, stride=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0))  # 256x17x17
    conv_3x3_reduce_bn = L.BatchNorm(conv_3x3_reduce, use_global_stats=False, in_place=True)
    conv_3x3_reduce_scale = L.Scale(conv_3x3_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_reduce_relu = L.ReLU(conv_3x3_reduce, in_place=True)

    conv_3x3 = L.Convolution(conv_3x3_reduce, kernel_size=3, num_output=384, stride=2,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 384x8x8
    conv_3x3_bn = L.BatchNorm(conv_3x3, use_global_stats=False, in_place=True)
    conv_3x3_scale = L.Scale(conv_3x3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_relu = L.ReLU(conv_3x3, in_place=True)

    conv_3x3_2_reduce = L.Convolution(bottom, kernel_size=1, num_output=256, stride=1,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))  # 256x17x17
    conv_3x3_2_reduce_bn = L.BatchNorm(conv_3x3_2_reduce, use_global_stats=False, in_place=True)
    conv_3x3_2_reduce_scale = L.Scale(conv_3x3_2_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_reduce_relu = L.ReLU(conv_3x3_2_reduce, in_place=True)

    conv_3x3_2 = L.Convolution(conv_3x3_2_reduce, kernel_size=3, num_output=256, stride=2,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 256x8x8
    conv_3x3_2_bn = L.BatchNorm(conv_3x3_2, use_global_stats=False, in_place=True)
    conv_3x3_2_scale = L.Scale(conv_3x3_2, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_2_relu = L.ReLU(conv_3x3_2, in_place=True)

    conv_3x3_3_reduce = L.Convolution(bottom, kernel_size=1, num_output=256, stride=1,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))  # 256x17x17
    conv_3x3_3_reduce_bn = L.BatchNorm(conv_3x3_3_reduce, use_global_stats=False, in_place=True)
    conv_3x3_3_reduce_scale = L.Scale(conv_3x3_3_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_3_reduce_relu = L.ReLU(conv_3x3_3_reduce, in_place=True)

    conv_3x3_3 = L.Convolution(conv_3x3_3_reduce, kernel_size=3, num_output=256, stride=1, pad=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 256x17x17
    conv_3x3_3_bn = L.BatchNorm(conv_3x3_3, use_global_stats=False, in_place=True)
    conv_3x3_3_scale = L.Scale(conv_3x3_3, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_3_relu = L.ReLU(conv_3x3_3, in_place=True)

    conv_3x3_4 = L.Convolution(conv_3x3_3, kernel_size=3, num_output=256, stride=2,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 256x8x8
    conv_3x3_4_bn = L.BatchNorm(conv_3x3_4, use_global_stats=False, in_place=True)
    conv_3x3_4_scale = L.Scale(conv_3x3_4, scale_param=dict(bias_term=True), in_place=True)
    conv_3x3_4_relu = L.ReLU(conv_3x3_4, in_place=True)

    concat = L.Concat(pool, conv_3x3, conv_3x3_2, conv_3x3_4)  # 2048x8x8

    return pool, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, \
           conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3_reduce, conv_3x3_3_reduce_bn, conv_3x3_3_reduce_scale, conv_3x3_3_reduce_relu, \
           conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, conv_3x3_4, conv_3x3_4_bn, conv_3x3_4_scale, \
           conv_3x3_4_relu, concat


def factorization_inception_resnet_c(bottom):
    """
    input:2048x8x8
    output:2048x8x8
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=192, stride=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 192x8x8
    conv_1x1_bn = L.BatchNorm(conv_1x1, use_global_stats=False, in_place=True)
    conv_1x1_scale = L.Scale(conv_1x1, scale_param=dict(bias_term=True), in_place=True)
    conv_1x1_relu = L.ReLU(conv_1x1, in_place=True)

    conv_1x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=192, stride=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0))  # 192x8x8
    conv_1x3_reduce_bn = L.BatchNorm(conv_1x3_reduce, use_global_stats=False, in_place=True)
    conv_1x3_reduce_scale = L.Scale(conv_1x3_reduce, scale_param=dict(bias_term=True), in_place=True)
    conv_1x3_reduce_relu = L.ReLU(conv_1x3_reduce, in_place=True)

    conv_1x3 = L.Convolution(conv_1x3_reduce, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1, stride=1, num_output=224,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 224x8x8
    conv_1x3_bn = L.BatchNorm(conv_1x3, use_global_stats=False, in_place=True)
    conv_1x3_scale = L.Scale(conv_1x3, scale_param=dict(bias_term=True), in_place=True)
    conv_1x3_relu = L.ReLU(conv_1x3, in_place=True)

    conv_3x1 = L.Convolution(conv_1x3, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0, stride=1, num_output=256,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0))  # 256x8x8
    conv_3x1_bn = L.BatchNorm(conv_3x1, use_global_stats=False, in_place=True)
    conv_3x1_scale = L.Scale(conv_3x1, scale_param=dict(bias_term=True), in_place=True)
    conv_3x1_relu = L.ReLU(conv_3x1, in_place=True)

    concat = L.Concat(conv_1x1, conv_3x1)  # 448x8x8

    conv_1x1_2 = L.Convolution(concat, kernel_size=1, num_output=2048, stride=1,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0))  # 2048x8x8
    conv_1x1_2_bn = L.BatchNorm(conv_1x1_2, use_global_stats=False, in_place=True)
    conv_1x1_2_scale = L.Scale(conv_1x1_2, scale_param=dict(bias_term=True), in_place=True)
    # conv_1x1_2_relu = L.ReLU(conv_1x1_2_scale, in_place=True)  # linear activation

    residual_eltwise = L.Eltwise(bottom, conv_1x1_2,
                                 eltwise_param=dict(operation=1))

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, conv_1x3_reduce, conv_1x3_reduce_bn, \
           conv_1x3_reduce_scale, conv_1x3_reduce_relu, conv_1x3, conv_1x3_bn, conv_1x3_scale, conv_1x3_relu, \
           conv_3x1, conv_3x1_bn, conv_3x1_scale, conv_3x1_relu, concat, conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, \
           residual_eltwise


inception_resnet_a = 'n.inception_resnet_a(order)_1x1, n.inception_resnet_a(order)_1x1_bn, n.inception_resnet_a(order)_1x1_scale, \
        n.inception_resnet_a(order)_1x1_relu, n.inception_resnet_a(order)_3x3_reduce, n.inception_resnet_a(order)_3x3_reduce_bn, \
        n.inception_resnet_a(order)_3x3_reduce_scale, n.inception_resnet_a(order)_3x3_reduce_relu, n.inception_resnet_a(order)_3x3, \
        n.inception_resnet_a(order)_3x3_bn, n.inception_resnet_a(order)_3x3_scale, n.inception_resnet_a(order)_3x3_relu, \
        n.inception_resnet_a(order)_3x3_2_reduce, n.inception_resnet_a(order)_3x3_2_reduce_bn, n.inception_resnet_a(order)_3x3_2_reduce_scale, \
        n.inception_resnet_a(order)_3x3_2_reduce_relu, n.inception_resnet_a(order)_3x3_2, n.inception_resnet_a(order)_3x3_2_bn, \
        n.inception_resnet_a(order)_3x3_2_scale, n.inception_resnet_a(order)_3x3_2_relu, n.inception_resnet_a(order)_3x3_3, \
        n.inception_resnet_a(order)_3x3_3_bn, n.inception_resnet_a(order)_3x3_3_scale, n.inception_resnet_a(order)_3x3_3_relu, \
        n.inception_resnet_a(order)_concat, n.inception_resnet_a(order)_1x1_2, n.inception_resnet_a(order)_1x1_2_bn, \
        n.inception_resnet_a(order)_1x1_2_scale, n.inception_resnet_a(order)_residual_eltwise = \
            factorization_inception_resnet_a(bottom)'

inception_resnet_b = 'n.inception_resnet_b(order)_1x1, n.inception_resnet_b(order)_1x1_bn, n.inception_resnet_b(order)_1x1_scale, \
        n.inception_resnet_b(order)_1x1_relu, n.inception_resnet_b(order)_1x7_reduce, n.inception_resnet_b(order)_1x7_reduce_bn, \
        n.inception_resnet_b(order)_1x7_reduce_scale, n.inception_resnet_b(order)_1x7_reduce_relu, n.inception_resnet_b(order)_1x7, \
        n.inception_resnet_b(order)_1x7_bn, n.inception_resnet_b(order)_1x7_scale, n.inception_resnet_b(order)_1x7_relu, \
        n.inception_resnet_b(order)_7x1, n.inception_resnet_b(order)_7x1_bn, n.inception_resnet_b(order)_7x1_scale, \
        n.inception_resnet_b(order)_7x1_relu, n.inception_resnet_b(order)_concat, n.inception_resnet_b(order)_1x1_2, \
        n.inception_resnet_b(order)_1x1_2_bn, n.inception_resnet_b(order)_1x1_2_scale, n.inception_resnet_b(order)_residual_eltwise \
            = factorization_inception_resnet_b(bottom)'

inception_resnet_c = 'n.inception_resnet_c(order)_1x1, n.inception_resnet_c(order)_1x1_bn, n.inception_resnet_c(order)_1x1_scale, \
        n.inception_resnet_c(order)_1x1_relu, n.inception_resnet_c(order)_1x3_reduce, n.inception_resnet_c(order)_1x3_reduce_bn, \
        n.inception_resnet_c(order)_1x3_reduce_scale, n.inception_resnet_c(order)_1x3_reduce_relu, n.inception_resnet_c(order)_1x3, \
        n.inception_resnet_c(order)_1x3_bn, n.inception_resnet_c(order)_1x3_scale, n.inception_resnet_c(order)_1x3_relu, \
        n.inception_resnet_c(order)_3x1, n.inception_resnet_c(order)_3x1_bn, n.inception_resnet_c(order)_3x1_scale, \
        n.inception_resnet_c(order)_3x1_relu, n.inception_resnet_c(order)_concat, n.inception_resnet_c(order)_1x1_2, \
        n.inception_resnet_c(order)_1x1_2_bn, n.inception_resnet_c(order)_1x1_2_scale, n.inception_resnet_c(order)_residual_eltwise = \
            factorization_inception_resnet_c(bottom)'


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
        n.conv1_3x3_s2, n.conv1_3x3_s2_bn, n.conv1_3x3_s2_scale, n.conv1_3x3_s2_relu, n.conv2_3x3_s1, n.conv2_3x3_s1_bn, \
        n.conv2_3x3_s1_scale, n.conv2_3x3_s1_relu, n.conv3_3x3_s1, n.conv3_3x3_s1_bn, n.conv3_3x3_s1_scale, n.conv3_3x3_s1_relu, \
        n.inception_stem1_3x3_s2, n.inception_stem1_3x3_s2_bn, n.inception_stem1_3x3_s2_scale, n.inception_stem1_3x3_s2_relu, \
        n.inception_stem1_pool, n.inception_stem1, n.inception_stem2_3x3_reduce, n.inception_stem2_3x3_reduce_bn, \
        n.inception_stem2_3x3_reduce_scale, n.inception_stem2_3x3_reduce_relu, n.inception_stem2_3x3, \
        n.inception_stem2_3x3_bn, n.inception_stem2_3x3_scale, n.inception_stem2_3x3_relu, n.inception_stem2_7x1_reduce, \
        n.inception_stem2_7x1_reduce_bn, n.inception_stem2_7x1_reduce_scale, n.inception_stem2_7x1_reduce_relu, \
        n.inception_stem2_7x1, n.inception_stem2_7x1_bn, n.inception_stem2_7x1_scale, n.inception_stem2_7x1_relu, \
        n.inception_stem2_1x7, n.inception_stem2_1x7_bn, n.inception_stem2_1x7_scale, n.inception_stem2_1x7_relu, \
        n.inception_stem2_3x3_2, n.inception_stem2_3x3_2_bn, n.inception_stem2_3x3_2_scale, n.inception_stem2_3x3_2_relu, \
        n.inception_stem2, n.inception_stem3_3x3_s2, n.inception_stem3_3x3_s2_bn, n.inception_stem3_3x3_s2_scale, \
        n.inception_stem3_3x3_s2_relu, n.inception_stem3_pool, n.inception_stem3 = \
            stem_299x299(n.data)  # 384x35x35

        # 5 x inception_resnet_a
        for i in xrange(5):
            if i == 0:
                bottom = 'n.inception_stem3'
            else:
                bottom = 'n.inception_resnet_a(order)_residual_eltwise'.replace('(order)', str(i))
            exec (inception_resnet_a.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 384x35x35

        # reduction_a
        n.reduction_a_pool, n.reduction_a_3x3, n.reduction_a_3x3_bn, n.reduction_a_3x3_scale, n.reduction_a_3x3_relu, \
        n.reduction_a_3x3_2_reduce, n.reduction_a_3x3_2_reduce_bn, n.reduction_a_3x3_2_reduce_scale, \
        n.reduction_a_3x3_2_reduce_relu, n.reduction_a_3x3_2, n.reduction_a_3x3_2_bn, n.reduction_a_3x3_2_scale, \
        n.reduction_a_3x3_2_relu, n.reduction_a_3x3_3, n.reduction_a_3x3_3_bn, n.reduction_a_3x3_3_scale, \
        n.reduction_a_3x3_3_relu, n.reduction_a_concat = \
            reduction_a(n.inception_resnet_a5_residual_eltwise)  # 1152x17x17

        # 10 x inception_resnet_b
        for i in xrange(10):
            if i == 0:
                bottom = 'n.reduction_a_concat'
            else:
                bottom = 'n.inception_resnet_b(order)_residual_eltwise'.replace('(order)', str(i))
            exec (inception_resnet_b.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 1152x17x17

        # reduction_b
        n.reduction_b_pool, n.reduction_b_3x3_reduce, n.reduction_b_3x3_reduce_bn, n.reduction_b_3x3_reduce_scale, \
        n.reduction_b_3x3_reduce_relu, n.reduction_b_3x3, n.reduction_b_3x3_bn, n.reduction_b_3x3_scale, \
        n.reduction_b_3x3_relu, n.reduction_b_3x3_2_reduce, n.reduction_b_3x3_2_reduce_bn, n.reduction_b_3x3_2_reduce_scale, \
        n.reduction_b_3x3_2_reduce_relu, n.reduction_b_3x3_2, n.reduction_b_3x3_2_bn, n.reduction_b_3x3_2_scale, \
        n.reduction_b_3x3_2_relu, n.reduction_b_3x3_3_reduce, n.reduction_b_3x3_3_reduce_bn, n.reduction_b_3x3_3_reduce_scale, \
        n.reduction_b_3x3_3_reduce_relu, n.reduction_b_3x3_3, n.reduction_b_3x3_3_bn, n.reduction_b_3x3_3_scale, \
        n.reduction_b_3x3_3_relu, n.reduction_b_3x3_4, n.reduction_b_3x3_4_bn, n.reduction_b_3x3_4_scale, \
        n.reduction_b_3x3_4_relu, n.reduction_b_concat = \
            reduction_b(n.inception_resnet_b10_residual_eltwise)  # 2048x8x8

        # 5 x inception_resnet_c
        for i in xrange(5):
            if i == 0:
                bottom = 'n.reduction_b_concat'
            else:
                bottom = 'n.inception_resnet_c(order)_residual_eltwise'.replace('(order)', str(i))
            exec (inception_resnet_c.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 2048x8x8

        n.pool_8x8_s1 = L.Pooling(n.inception_resnet_c5_residual_eltwise,
                                  pool=P.Pooling.AVE,
                                  global_pooling=True)  # 2048x1x1
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
