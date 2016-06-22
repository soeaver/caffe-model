import caffe
from caffe import layers as L
from caffe import params as P


def conv_bn_scale_relu(bottom, num_output=64, kernel_size=1, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=1, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def bn_scale_relu(bottom):
    bn = L.BatchNorm(bottom, use_global_stats=False)
    scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(bn, in_place=True)

    return bn, scale, relu


def branch(bottom, num_output=64):
    bn0, scale0, relu0 = bn_scale_relu(bottom)
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu(bn0, num_output=num_output)
    conv2, bn2, scale2, relu2 = conv_bn_scale_relu(conv1, num_output=num_output, kernel_size=3, pad=1)
    conv3 = L.Convolution(conv2, num_output=num_output * 4, kernel_size=1, stride=1, pad=0)

    return bn0, scale0, relu0, conv1, bn1, scale1, relu1, conv2, bn2, scale2, relu2, conv3


def fractal_block(bottom, base_output=64):
    conv1a, bn1a, scale1a = conv_bn_scale(bottom, num_output=base_output * 4)
    bn1b0, scale1b0, relu1b0, conv1b1, bn1b1, scale1b1, relu1b1, conv1b2, bn1b2, scale1b2, relu1b2, conv1b3 = \
        branch(bottom, num_output=base_output)
    eltwise1 = L.Eltwise(conv1a, conv1b3, eltwise_param=dict(operation=1))

    conv2a, bn2a, scale2a = conv_bn_scale(eltwise1, num_output=base_output * 4)
    bn2b0, scale2b0, relu2b0, conv2b1, bn2b1, scale2b1, relu2b1, conv2b2, bn2b2, scale2b2, relu2b2, conv2b3 = \
        branch(eltwise1, num_output=base_output)
    conv12a, bn12a, scale12a = conv_bn_scale(bottom, num_output=base_output * 4)
    eltwise2 = L.Eltwise(conv2a, conv2b3, conv12a, eltwise_param=dict(operation=1))

    conv3a, bn3a, scale3a = conv_bn_scale(eltwise2, num_output=base_output * 4)
    bn3b0, scale3b0, relu3b0, conv3b1, bn3b1, scale3b1, relu3b1, conv3b2, bn3b2, scale3b2, relu3b2, conv3b3 = \
        branch(eltwise2, num_output=base_output)
    eltwise3 = L.Eltwise(conv3a, conv3b3, eltwise_param=dict(operation=1))

    conv4a, bn4a, scale4a = conv_bn_scale(eltwise3, num_output=base_output * 4)
    bn4b0, scale4b0, relu4b0, conv4b1, bn4b1, scale4b1, relu4b1, conv4b2, bn4b2, scale4b2, relu4b2, conv4b3 = \
        branch(eltwise3, num_output=base_output)
    conv34a, bn34a, scale34a = conv_bn_scale(eltwise2, num_output=base_output * 4)
    conv1234a, bn1234a, scale1234a = conv_bn_scale(bottom, num_output=base_output * 4)
    eltwise4 = L.Eltwise(conv4a, conv4b3, conv34a, conv1234a, eltwise_param=dict(operation=1))

    return conv1a, bn1a, scale1a, bn1b0, scale1b0, relu1b0, conv1b1, bn1b1, scale1b1, relu1b1, conv1b2, bn1b2, \
           scale1b2, relu1b2, conv1b3, eltwise1, conv2a, bn2a, scale2a, bn2b0, scale2b0, relu2b0, conv2b1, bn2b1, \
           scale2b1, relu2b1, conv2b2, bn2b2, scale2b2, relu2b2, conv2b3, conv12a, bn12a, scale12a, eltwise2, \
           conv3a, bn3a, scale3a, bn3b0, scale3b0, relu3b0, conv3b1, bn3b1, scale3b1, relu3b1, conv3b2, bn3b2, \
           scale3b2, relu3b2, conv3b3, eltwise3, conv4a, bn4a, scale4a, bn4b0, scale4b0, relu4b0, conv4b1, bn4b1, \
           scale4b1, relu4b1, conv4b2, bn4b2, scale4b2, relu4b2, conv4b3, conv34a, bn34a, scale34a, conv1234a, \
           bn1234a, scale1234a, eltwise4


fractal_string = 'n.fractal_(stage)(order)_conv1a, n.fractal_(stage)(order)_bn1a, n.fractal_(stage)(order)_scale1a, \
    n.fractal_(stage)(order)_bn1b0, n.fractal_(stage)(order)_scale1b0, n.fractal_(stage)(order)_relu1b0, \
    n.fractal_(stage)(order)_conv1b1, n.fractal_(stage)(order)_bn1b1, n.fractal_(stage)(order)_scale1b1, \
    n.fractal_(stage)(order)_relu1b1, n.fractal_(stage)(order)_conv1b2, n.fractal_(stage)(order)_bn1b2, \
    n.fractal_(stage)(order)_scale1b2, n.fractal_(stage)(order)_relu1b2, n.fractal_(stage)(order)_conv1b3, \
    n.fractal_(stage)(order)_eltwise1, n.fractal_(stage)(order)_conv2a, n.fractal_(stage)(order)_bn2a, \
    n.fractal_(stage)(order)_scale2a, n.fractal_(stage)(order)_bn2b0, n.fractal_(stage)(order)_scale2b0, \
    n.fractal_(stage)(order)_relu2b0, n.fractal_(stage)(order)_conv2b1, n.fractal_(stage)(order)_bn2b1, \
    n.fractal_(stage)(order)_scale2b1, n.fractal_(stage)(order)_relu2b1, n.fractal_(stage)(order)_conv2b2, \
    n.fractal_(stage)(order)_bn2b2, n.fractal_(stage)(order)_scale2b2, n.fractal_(stage)(order)_relu2b2, \
    n.fractal_(stage)(order)_conv2b3, n.fractal_(stage)(order)_conv12a, n.fractal_(stage)(order)_bn12a, \
    n.fractal_(stage)(order)_scale12a, n.fractal_(stage)(order)_eltwise2, n.fractal_(stage)(order)_conv3a, \
    n.fractal_(stage)(order)_bn3a, n.fractal_(stage)(order)_scale3a, n.fractal_(stage)(order)_bn3b0, \
    n.fractal_(stage)(order)_scale3b0, n.fractal_(stage)(order)_relu3b0, n.fractal_(stage)(order)_conv3b1, \
    n.fractal_(stage)(order)_bn3b1, n.fractal_(stage)(order)_scale3b1, n.fractal_(stage)(order)_relu3b1, \
    n.fractal_(stage)(order)_conv3b2, n.fractal_(stage)(order)_bn3b2, n.fractal_(stage)(order)_scale3b2, \
    n.fractal_(stage)(order)_relu3b2, n.fractal_(stage)(order)_conv3b3, n.fractal_(stage)(order)_eltwise3, \
    n.fractal_(stage)(order)_conv4a, n.fractal_(stage)(order)_bn4a, n.fractal_(stage)(order)_scale4a, \
    n.fractal_(stage)(order)_bn4b0, n.fractal_(stage)(order)_scale4b0, n.fractal_(stage)(order)_relu4b0, \
    n.fractal_(stage)(order)_conv4b1, n.fractal_(stage)(order)_bn4b1, n.fractal_(stage)(order)_scale4b1, \
    n.fractal_(stage)(order)_relu4b1, n.fractal_(stage)(order)_conv4b2, n.fractal_(stage)(order)_bn4b2, \
    n.fractal_(stage)(order)_scale4b2, n.fractal_(stage)(order)_relu4b2, n.fractal_(stage)(order)_conv4b3, \
    n.fractal_(stage)(order)_conv34a, n.fractal_(stage)(order)_bn34a, n.fractal_(stage)(order)_scale34a, \
    n.fractal_(stage)(order)_conv1234a, n.fractal_(stage)(order)_bn1234a, n.fractal_(stage)(order)_scale1234a, \
    n.fractal_(stage)(order)_eltwise4 = fractal_block((bottom), base_output=(num))'

maxpool_string = 'n.pool(order) = L.Pooling((bottom), kernel_size=3, stride=2, pool=P.Pooling.MAX)'


class FractalNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def fractalnet_layers_proto(self, batch_size, phase='TRAIN', stages=(1, 1, 2, 1)):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
            conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3)  # 64x112x112
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56

        for num in xrange(len(stages)):  # num = 0, 1, 2, 3
            exec (maxpool_string.replace('(order)', str(num + 1))
                  .replace('(bottom)', ['n.conv1', 'n.fractal_a%s_eltwise4' % str(stages[0]),
                                        'n.fractal_b%s_eltwise4' % str(stages[1]),
                                        'n.fractal_c%s_eltwise4' % str(stages[2])][num]))
            for i in xrange(stages[num]):
                exec (fractal_string.replace('(stage)', 'abcd'[num])
                      .replace('(order)', str(i + 1))
                      .replace('(num)', str(2 ** num * 64))
                      .replace('(bottom)', ['n.pool%s' % str(num + 1),
                                            'n.fractal_%s%s_eltwise4' % ('abcd'[num], str(i))][0 < i]))

        exec 'n.pool5 = L.Pooling((bottom), pool=P.Pooling.AVE, global_pooling=True)'.\
            replace('(bottom)', 'n.fractal_d%s_eltwise4' % str(stages[3]))
        n.classifier = L.InnerProduct(n.pool5, num_output=self.classifier_num,
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
