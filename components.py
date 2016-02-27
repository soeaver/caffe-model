from caffe import layers as L
from caffe import params as P


def conv_relu_lrn_pool(bottom, conv_param, pool_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    relu = L.ReLU(conv, in_place=True)
    norm = L.LRN(conv, local_size=5, alpha=1e-4, beta=0.75)
    pool = L.Pooling(norm, pool=pool_param['type'], kernel_size=pool_param['kernel_size'],
                     stride=pool_param['stride'], pad=pool_param['pad'])
    return conv, relu, norm, pool


def conv_relu_pool_lrn(bottom, conv_param, pool_param):
    conv = L.Convolution(bottom, num_output=conv_param['num_output'], kernel_size=conv_param['kernel_size'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    relu = L.ReLU(conv, in_place=True)
    pool = L.Pooling(conv, pool=pool_param['type'], kernel_size=pool_param['kernel_size'],
                     stride=pool_param['stride'], pad=pool_param['pad'])
    norm = L.LRN(pool, local_size=5, alpha=1e-4, beta=0.75)

    return conv, relu, pool, norm


def conv_relu_pool(bottom, conv_param, pool_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    relu = L.ReLU(conv, in_place=True)
    pool = L.Pooling(conv, pool=pool_param['type'], kernel_size=pool_param['kernel_size'],
                     stride=pool_param['stride'], pad=pool_param['pad'])
    return conv, relu, pool


def conv_bn_relu_pool(bottom, conv_param, pool_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    bn = L.BatchNorm(conv, use_global_stats=False)
    relu = L.ReLU(bn, in_place=True)
    pool = L.Pooling(bn, pool=pool_param['type'], kernel_size=pool_param['kernel_size'],
                     stride=pool_param['stride'], pad=pool_param['pad'])
    return conv, bn, relu, pool


def conv_pool_bn_relu(bottom, conv_param, pool_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    pool = L.Pooling(conv, pool=pool_param['type'], kernel_size=pool_param['kernel_size'],
                     stride=pool_param['stride'], pad=pool_param['pad'])
    bn = L.BatchNorm(pool, use_global_stats=False)
    relu = L.ReLU(bn, in_place=True)

    return conv, pool, bn, relu


def conv_bn_relu(bottom, conv_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    bn = L.BatchNorm(conv, use_global_stats=False)
    relu = L.ReLU(bn, in_place=True)
    return conv, bn, relu


def conv_relu(bottom, conv_param):
    conv = L.Convolution(bottom, kernel_size=conv_param['kernel_size'], num_output=conv_param['num_output'],
                         stride=conv_param['stride'], pad=conv_param['pad'], group=conv_param['group'],
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type=conv_param['weight_type'], std=conv_param['weight_std']),
                         bias_filler=dict(type=conv_param['bias_type'], value=conv_param['bias_value']))
    relu = L.ReLU(conv, in_place=True)
    return conv, relu


def fc_relu_drop(bottom, fc_param, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=fc_param['num_output'],
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type=fc_param['weight_type'], std=fc_param['weight_std']),
                        bias_filler=dict(type='constant', value=fc_param['bias_value']))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def fc_relu(bottom, fc_param):
    fc = L.InnerProduct(bottom, num_output=fc_param['num_output'],
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type=fc_param['weight_type'], std=fc_param['weight_std']),
                        bias_filler=dict(type='constant', value=fc_param['bias_value']))
    relu = L.ReLU(fc, in_place=True)
    return fc, relu


def accuracy_top1_top5(bottom, label):
    accuracy_top1 = L.Accuracy(bottom, label, include=dict(phase=1))
    accuracy_top5 = L.Accuracy(bottom, label, include=dict(phase=1),
                               accuracy_param=dict(top_k=5))
    return accuracy_top1, accuracy_top5


def inception(bottom, conv_output):
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_1x1'],
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    relu_1x1 = L.ReLU(conv_1x1, in_place=True)

    conv_3x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_3x3_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0.2))
    relu_3x3_reduce = L.ReLU(conv_3x3_reduce, in_place=True)
    conv_3x3 = L.Convolution(conv_3x3_reduce, kernel_size=3, num_output=conv_output['conv_3x3'], pad=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    relu_3x3 = L.ReLU(conv_3x3, in_place=True)

    conv_5x5_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_5x5_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0.2))
    relu_5x5_reduce = L.ReLU(conv_5x5_reduce, in_place=True)
    conv_5x5 = L.Convolution(conv_5x5_reduce, kernel_size=5, num_output=conv_output['conv_5x5'], pad=2,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    relu_5x5 = L.ReLU(conv_5x5, in_place=True)

    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    pool_proj = L.Convolution(pool, kernel_size=1, num_output=conv_output['pool_proj'],
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type='xavier'),
                              bias_filler=dict(type='constant', value=0.2))
    relu_pool_proj = L.ReLU(pool_proj, in_place=True)
    concat = L.Concat(conv_1x1, conv_3x3, conv_5x5, pool_proj)

    return conv_1x1, relu_1x1, conv_3x3_reduce, relu_3x3_reduce, conv_3x3, relu_3x3, conv_5x5_reduce, relu_5x5_reduce, \
           conv_5x5, relu_5x5, pool, pool_proj, relu_pool_proj, concat


def inception_bn(bottom, conv_output):
    conv_1x1 = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_1x1'],
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    bn_1x1 = L.BatchNorm(conv_1x1, use_global_stats=False)
    relu_1x1 = L.ReLU(bn_1x1, in_place=True)

    conv_3x3_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_3x3_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0.2))
    bn_3x3_reduce = L.BatchNorm(conv_3x3_reduce, use_global_stats=False)
    relu_3x3_reduce = L.ReLU(bn_3x3_reduce, in_place=True)
    conv_3x3 = L.Convolution(bn_3x3_reduce, kernel_size=3, num_output=conv_output['conv_3x3'], pad=1,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    bn_3x3 = L.BatchNorm(conv_3x3, use_global_stats=False)
    relu_3x3 = L.ReLU(bn_3x3, in_place=True)

    conv_5x5_reduce = L.Convolution(bottom, kernel_size=1, num_output=conv_output['conv_5x5_reduce'],
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'),
                                    bias_filler=dict(type='constant', value=0.2))
    bn_5x5_reduce = L.BatchNorm(conv_5x5_reduce, use_global_stats=False)
    relu_5x5_reduce = L.ReLU(bn_5x5_reduce, in_place=True)
    conv_5x5 = L.Convolution(bn_5x5_reduce, kernel_size=5, num_output=conv_output['conv_5x5'], pad=2,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant', value=0.2))
    bn_5x5 = L.BatchNorm(conv_5x5, use_global_stats=False)
    relu_5x5 = L.ReLU(bn_5x5, in_place=True)

    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    pool_proj = L.Convolution(pool, kernel_size=1, num_output=conv_output['pool_proj'],
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type='xavier'),
                              bias_filler=dict(type='constant', value=0.2))
    bn_proj = L.BatchNorm(pool_proj, use_global_stats=False)
    relu_pool_proj = L.ReLU(bn_proj, in_place=True)
    concat = L.Concat(bn_1x1, bn_3x3, bn_5x5, bn_proj)

    return conv_1x1, bn_1x1, relu_1x1, conv_3x3_reduce, bn_3x3_reduce, relu_3x3_reduce, conv_3x3, bn_3x3, relu_3x3, \
           conv_5x5_reduce, bn_5x5_reduce, relu_5x5_reduce, conv_5x5, bn_5x5, relu_5x5, pool, pool_proj, bn_proj, \
           relu_pool_proj, concat


def inception_v3_7a(bottom, conv_output):
    conv_1x1, conv_1x1_bn, conv_1x1_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_1x1'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_5x5_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_5x5, conv_5x5_bn, conv_5x5_relu = \
        conv_bn_relu(conv_5x5_reduce_bn, dict(kernel_size=5, num_output=conv_output['conv_5x5'], stride=1, pad=2,
                                              group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                              bias_value=0))
    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_3x3_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu = \
        conv_bn_relu(conv_3x3_reduce_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_1'], stride=1, pad=1,
                                              group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                              bias_value=0))
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_relu = \
        conv_bn_relu(conv_3x3_1_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_2'], stride=1, pad=1, group=1,
                                         weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)
    pool_proj, pool_proj_bn, pool_proj_relu = \
        conv_bn_relu(pool, dict(kernel_size=1, num_output=conv_output['pool_proj'], stride=1, pad=0, group=1,
                                weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    concat = L.Concat(conv_1x1_bn, conv_5x5_bn, conv_3x3_2_bn, pool_proj_bn)

    return conv_1x1, conv_1x1_bn, conv_1x1_relu, conv_5x5_reduce, conv_5x5_reduce_bn, conv_5x5_reduce_relu, \
           conv_5x5, conv_5x5_bn, conv_5x5_relu, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu, \
           conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_relu, pool, \
           pool_proj, pool_proj_bn, pool_proj_relu, concat


def inception_v3_7b(bottom, conv_output):
    conv_3x3_0, conv_3x3_0_bn, conv_3x3_0_relu = \
        conv_bn_relu(bottom, dict(kernel_size=3, num_output=conv_output['conv_3x3_0'], stride=2, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_3x3_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu = \
        conv_bn_relu(conv_3x3_reduce_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_1'], stride=1, pad=1,
                                              group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                              bias_value=0))
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_relu = \
        conv_bn_relu(conv_3x3_1_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_2'], stride=2, pad=0, group=1,
                                         weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pad=0, pool=P.Pooling.MAX)
    # pool_proj, pool_proj_bn, pool_proj_relu = \
    #     conv_bn_relu(pool, dict(kernel_size=1, num_output=conv_output['pool_proj'], stride=1, pad=0, group=1,
    #                             weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    concat = L.Concat(conv_3x3_0_bn, conv_3x3_2_bn, pool)

    return conv_3x3_0, conv_3x3_0_bn, conv_3x3_0_relu, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu, \
           conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_relu, pool, concat
           

def factorization_conv(bottom, kernel=(1, 7), pad=(0, 3), output=(128, 192)):
    conv_1 = L.Convolution(bottom, kernel_h=kernel[0], kernel_w=kernel[1], pad_h=pad[0], pad_w=pad[1], stride=1,
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                           weight_filler=dict(type='xavier'), num_output=output[0],
                           bias_filler=dict(type='constant', value=0))
    conv_1_bn = L.BatchNorm(conv_1, use_global_stats=False)
    conv_1_relu = L.ReLU(conv_1_bn, in_place=True)
    conv_2 = L.Convolution(conv_1_bn, kernel_h=kernel[1], kernel_w=kernel[0], pad_h=pad[1], pad_w=pad[0], stride=1,
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                           weight_filler=dict(type='xavier'), num_output=output[1],
                           bias_filler=dict(type='constant', value=0))
    conv_2_bn = L.BatchNorm(conv_2, use_global_stats=False)
    conv_2_relu = L.ReLU(conv_2_bn, in_place=True)

    return conv_1, conv_1_bn, conv_1_relu, conv_2, conv_2_bn, conv_2_relu


def inception_v3_7c(bottom, conv_output):
    conv_1x1, conv_1x1_bn, conv_1x1_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_1x1'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_1x7_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_1x7_0, conv_1x7_0_bn, conv_1x7_0_relu, conv_7x1_0, conv_7x1_0_bn, conv_7x1_0_relu = \
        factorization_conv(conv_1x7_reduce_bn, output=(conv_output['conv_1x7_0'], conv_output['conv_7x1_0']))
    conv_7x1_reduce, conv_7x1_reduce_bn, conv_7x1_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_7x1_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_7x1_1, conv_7x1_1_bn, conv_7x1_1_relu, conv_1x7_1, conv_1x7_1_bn, conv_1x7_1_relu = \
        factorization_conv(conv_7x1_reduce_bn, kernel=(7, 1), pad=(3, 0),
                           output=(conv_output['conv_1x7_1'], conv_output['conv_7x1_1']))
    conv_7x1_2, conv_7x1_2_bn, conv_7x1_2_relu, conv_1x7_2, conv_1x7_2_bn, conv_1x7_2_relu = \
        factorization_conv(conv_1x7_1_bn, kernel=(7, 1), pad=(3, 0),
                           output=(conv_output['conv_1x7_2'], conv_output['conv_7x1_2']))
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)
    pool_proj, pool_proj_bn, pool_proj_relu = \
        conv_bn_relu(pool, dict(kernel_size=1, num_output=conv_output['pool_proj'], stride=1, pad=0, group=1,
                                weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    concat = L.Concat(conv_1x1_bn, conv_7x1_0_bn, conv_1x7_2_bn, pool_proj_bn)

    return conv_1x1, conv_1x1_bn, conv_1x1_relu, conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_relu, \
           conv_1x7_0, conv_1x7_0_bn, conv_1x7_0_relu, conv_7x1_0, conv_7x1_0_bn, conv_7x1_0_relu, conv_7x1_reduce, \
           conv_7x1_reduce_bn, conv_7x1_reduce_relu, conv_7x1_1, conv_7x1_1_bn, conv_7x1_1_relu, conv_1x7_1, \
           conv_1x7_1_bn, conv_1x7_1_relu, conv_7x1_2, conv_7x1_2_bn, conv_7x1_2_relu, conv_1x7_2, conv_1x7_2_bn, \
           conv_1x7_2_relu, pool, pool_proj, pool_proj_bn, pool_proj_relu, concat


def inception_v3_7d(bottom, conv_output):
    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_3x3_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_0, conv_3x3_0_bn, conv_3x3_0_relu = \
        conv_bn_relu(conv_3x3_reduce_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_0'], stride=2, pad=0,
                                              group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                              bias_value=0))
    conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_1x7_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_1x7, conv_1x7_bn, conv_1x7_relu, conv_7x1, conv_7x1_bn, conv_7x1_relu = \
        factorization_conv(conv_1x7_reduce_bn, output=(conv_output['conv_1x7'], conv_output['conv_7x1']))
    conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu = \
        conv_bn_relu(conv_7x1_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_1'], stride=2, pad=0,
                                       group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                       bias_value=0))
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pad=0, pool=P.Pooling.MAX)
    concat = L.Concat(conv_3x3_0_bn, conv_3x3_1_bn, pool)

    return conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_relu, conv_3x3_0, conv_3x3_0_bn, conv_3x3_0_relu, \
           conv_1x7_reduce, conv_1x7_reduce_bn, conv_1x7_reduce_relu, conv_1x7, conv_1x7_bn, conv_1x7_relu, conv_7x1, \
           conv_7x1_bn, conv_7x1_relu, conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu, pool, concat


def inception_v3_7e(bottom, conv_output):
    conv_1x1, conv_1x1_bn, conv_1x1_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_1x1'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_0_reduce, conv_3x3_0_reduce_bn, conv_3x3_0_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_3x3_0_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_1x3_0, conv_1x3_0_bn, conv_1x3_0_relu, conv_3x1_0, conv_3x1_0_bn, conv_3x1_0_relu = \
        factorization_conv(conv_3x3_0_reduce_bn, kernel=(1, 3), pad=(0, 1),
                           output=(conv_output['conv_1x3_0'], conv_output['conv_3x1_0']))
    conv_3x3_1_reduce, conv_3x3_1_reduce_bn, conv_3x3_1_reduce_relu = \
        conv_bn_relu(bottom, dict(kernel_size=1, num_output=conv_output['conv_3x3_1_reduce'], stride=1, pad=0, group=1,
                                  weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu = \
        conv_bn_relu(conv_3x3_1_reduce_bn, dict(kernel_size=3, num_output=conv_output['conv_3x3_1'], stride=1, pad=1,
                                                group=1, weight_type='xavier', weight_std=0.01, bias_type='constant',
                                                bias_value=0))
    conv_1x3_1, conv_1x3_1_bn, conv_1x3_1_relu, conv_3x1_1, conv_3x1_1_bn, conv_3x1_1_relu = \
        factorization_conv(conv_3x3_1_bn, kernel=(1, 3), pad=(0, 1),
                           output=(conv_output['conv_1x3_1'], conv_output['conv_3x1_1']))
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=conv_output['pooling'])
    pool_proj, pool_proj_bn, pool_proj_relu = \
        conv_bn_relu(pool, dict(kernel_size=1, num_output=conv_output['pool_proj'], stride=1, pad=0, group=1,
                                weight_type='xavier', weight_std=0.01, bias_type='constant', bias_value=0))
    concat = L.Concat(conv_1x1_bn, conv_3x1_0_bn, conv_3x1_1_bn, pool_proj_bn)

    return conv_1x1, conv_1x1_bn, conv_1x1_relu, conv_3x3_0_reduce, conv_3x3_0_reduce_bn, conv_3x3_0_reduce_relu, \
           conv_1x3_0, conv_1x3_0_bn, conv_1x3_0_relu, conv_3x1_0, conv_3x1_0_bn, conv_3x1_0_relu, conv_3x3_1_reduce, \
           conv_3x3_1_reduce_bn, conv_3x3_1_reduce_relu, conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_relu, conv_1x3_1, \
           conv_1x3_1_bn, conv_1x3_1_relu, conv_3x1_1, conv_3x1_1_bn, conv_3x1_1_relu, pool, pool_proj, pool_proj_bn, \
           pool_proj_relu, concat


def conv_stack_3(bottom, conv_param):
    conv_1, relu_1 = conv_relu(bottom,
                               dict(num_output=conv_param['num_output'][0], kernel_size=conv_param['kernel_size'][0],
                                    stride=conv_param['stride'][0], pad=conv_param['pad'][0],
                                    group=conv_param['group'][0],
                                    weight_type=conv_param['weight_type'][0], weight_std=conv_param['weight_std'][0],
                                    bias_type=conv_param['bias_type'][0], bias_value=conv_param['bias_value'][0]))
    conv_2, relu_2 = conv_relu(conv_1,
                               dict(num_output=conv_param['num_output'][1], kernel_size=conv_param['kernel_size'][1],
                                    stride=conv_param['stride'][1], pad=conv_param['pad'][1],
                                    group=conv_param['group'][1],
                                    weight_type=conv_param['weight_type'][1], weight_std=conv_param['weight_std'][1],
                                    bias_type=conv_param['bias_type'][1], bias_value=conv_param['bias_value'][1]))
    conv_3, relu_3 = conv_relu(conv_2,
                               dict(num_output=conv_param['num_output'][2], kernel_size=conv_param['kernel_size'][2],
                                    stride=conv_param['stride'][2], pad=conv_param['pad'][2],
                                    group=conv_param['group'][2],
                                    weight_type=conv_param['weight_type'][2], weight_std=conv_param['weight_std'][2],
                                    bias_type=conv_param['bias_type'][2], bias_value=conv_param['bias_value'][2]))

    return conv_1, relu_1, conv_2, relu_2, conv_3, relu_3


def conv_bn_stack_3(bottom, conv_param):
    conv_1, bn_1, relu_1 = conv_bn_relu(bottom,
                                        dict(num_output=conv_param['num_output'][0],
                                             kernel_size=conv_param['kernel_size'][0], stride=conv_param['stride'][0],
                                             pad=conv_param['pad'][0], group=conv_param['group'][0],
                                             weight_type=conv_param['weight_type'][0],
                                             weight_std=conv_param['weight_std'][0],
                                             bias_type=conv_param['bias_type'][0],
                                             bias_value=conv_param['bias_value'][0]))
    conv_2, bn_2, relu_2 = conv_bn_relu(bn_1,
                                        dict(num_output=conv_param['num_output'][1],
                                             kernel_size=conv_param['kernel_size'][1], stride=conv_param['stride'][1],
                                             pad=conv_param['pad'][1], group=conv_param['group'][1],
                                             weight_type=conv_param['weight_type'][1],
                                             weight_std=conv_param['weight_std'][1],
                                             bias_type=conv_param['bias_type'][1],
                                             bias_value=conv_param['bias_value'][1]))
    conv_3, bn_3, relu_3 = conv_bn_relu(bn_2,
                                        dict(num_output=conv_param['num_output'][2],
                                             kernel_size=conv_param['kernel_size'][2], stride=conv_param['stride'][2],
                                             pad=conv_param['pad'][2], group=conv_param['group'][2],
                                             weight_type=conv_param['weight_type'][2],
                                             weight_std=conv_param['weight_std'][2],
                                             bias_type=conv_param['bias_type'][2],
                                             bias_value=conv_param['bias_value'][2]))

    return conv_1, bn_1, relu_1, conv_2, bn_2, relu_2, conv_3, bn_3, relu_3


def conv_stack_2(bottom, conv_param):
    conv_1, relu_1 = conv_relu(bottom,
                               dict(num_output=conv_param['num_output'][0], kernel_size=conv_param['kernel_size'][0],
                                    stride=conv_param['stride'][0], pad=conv_param['pad'][0],
                                    group=conv_param['group'][0],
                                    weight_type=conv_param['weight_type'][0], weight_std=conv_param['weight_std'][0],
                                    bias_type=conv_param['bias_type'][0], bias_value=conv_param['bias_value'][0]))
    conv_2, relu_2 = conv_relu(conv_1,
                               dict(num_output=conv_param['num_output'][1], kernel_size=conv_param['kernel_size'][1],
                                    stride=conv_param['stride'][1], pad=conv_param['pad'][1],
                                    group=conv_param['group'][1],
                                    weight_type=conv_param['weight_type'][1], weight_std=conv_param['weight_std'][1],
                                    bias_type=conv_param['bias_type'][1], bias_value=conv_param['bias_value'][1]))

    return conv_1, relu_1, conv_2, relu_2


def conv_bn_stack_2(bottom, conv_param):
    conv_1, bn_1, relu_1 = conv_bn_relu(bottom,
                                        dict(num_output=conv_param['num_output'][0],
                                             kernel_size=conv_param['kernel_size'][0], stride=conv_param['stride'][0],
                                             pad=conv_param['pad'][0], group=conv_param['group'][0],
                                             weight_type=conv_param['weight_type'][0],
                                             weight_std=conv_param['weight_std'][0],
                                             bias_type=conv_param['bias_type'][0],
                                             bias_value=conv_param['bias_value'][0]))
    conv_2, bn_2, relu_2 = conv_bn_relu(bn_1,
                                        dict(num_output=conv_param['num_output'][1],
                                             kernel_size=conv_param['kernel_size'][1], stride=conv_param['stride'][1],
                                             pad=conv_param['pad'][1], group=conv_param['group'][1],
                                             weight_type=conv_param['weight_type'][1],
                                             weight_std=conv_param['weight_std'][1],
                                             bias_type=conv_param['bias_type'][1],
                                             bias_value=conv_param['bias_value'][1]))

    return conv_1, bn_1, relu_1, conv_2, bn_2, relu_2
    

def spp(bottom, pool1_param, pool2_param, pool3_param):
    pool1 = L.Pooling(bottom, pool=pool1_param['type'], kernel_size=pool1_param['kernel_size'],
                      stride=pool1_param['stride'], pad=pool1_param['pad'])  # (MAX, 3, 1, 0)
    pool2 = L.Pooling(bottom, pool=pool2_param['type'], kernel_size=pool2_param['kernel_size'],
                      stride=pool2_param['stride'], pad=pool2_param['pad'])  # (MAX, 3, 2, 0)
    pool3 = L.Pooling(bottom, pool=pool3_param['type'], kernel_size=pool3_param['kernel_size'],
                      stride=pool3_param['stride'], pad=pool3_param['pad'])  # (MAX, 5, 5, 0)
    flatdata1 = L.Flatten(pool1)
    flatdata2 = L.Flatten(pool2)
    flatdata3 = L.Flatten(pool3)
    concat = L.Concat(flatdata1, flatdata2, flatdata3)
    return pool1, pool2, pool3, flatdata1, flatdata2, flatdata3, concat
