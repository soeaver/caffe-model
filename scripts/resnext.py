import caffe

from caffe import layers as L
from caffe import params as P


def resnext_block(bottom, base_output=64, card=32):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers

    Args:
        card:
    """
    conv1 = L.Convolution(bottom, num_output=base_output * (card / 16), kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output * (card / 16), kernel_size=3, stride=1, pad=1, group=card,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output * 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(bottom, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, eltwise, eltwise_relu


def match_block(bottom, base_output=64, stride=2, card=32):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    conv1 = L.Convolution(bottom, num_output=base_output * (card / 16), kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output * (card / 16), kernel_size=3, stride=stride, pad=1, group=card,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output * 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    match = L.Convolution(bottom, num_output=base_output * 4, kernel_size=1, stride=stride, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    match_bn = L.BatchNorm(match, use_global_stats=False, in_place=True)
    match_scale = L.Scale(match, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(match, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, match, match_bn, match_scale, eltwise, eltwise_relu


resnext_string = 'n.resx(n)_conv1, n.resx(n)_conv1_bn, n.resx(n)_conv1_scale, n.resx(n)_conv1_relu, \
        n.resx(n)_conv2, n.resx(n)_conv2_bn, n.resx(n)_conv2_scale, n.resx(n)_conv2_relu, n.resx(n)_conv3, \
        n.resx(n)_conv3_bn, n.resx(n)_conv3_scale, n.resx(n)_elewise, n.resx(n)_elewise_relu = \
            resnext_block((bottom), base_output=(base), card=(c))'

match_string = 'n.resx(n)_conv1, n.resx(n)_conv1_bn, n.resx(n)_conv1_scale, n.resx(n)_conv1_relu, \
    n.resx(n)_conv2, n.resx(n)_conv2_bn, n.resx(n)_conv2_scale, n.resx(n)_conv2_relu, n.resx(n)_conv3, \
    n.resx(n)_conv3_bn, n.resx(n)_conv3_scale, n.resx(n)_match_conv, n.resx(n)_match_conv_bn, n.resx(n)_match_conv_scale,\
    n.resx(n)_elewise, n.resx(n)_elewise_relu = match_block((bottom), base_output=(base), stride=(s), card=(c))'


class ResNeXt(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def resnext_layers_proto(self, batch_size, card=32, phase='TRAIN', stages=(3, 4, 6, 3)):
        """

        :param batch_size: the batch_size of train and test phase
        :param phase: TRAIN or TEST
        :param stages: the num of layers = 2 + 3*sum(stages), layers would better be chosen from [50, 101, 152]
                       {every stage is composed of 1 residual_branch_shortcut module and stage[i]-1 residual_branch
                       modules, each module consists of 3 conv layers}
                        (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
        """
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1 = L.Convolution(n.data, num_output=64, kernel_size=7, stride=2, pad=3, bias_term=False,
                                param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
        n.conv1_bn = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True)
        n.conv1_scale = L.Scale(n.conv1, scale_param=dict(bias_term=True), in_place=True)
        n.conv1_relu = L.ReLU(n.conv1, in_place=True)  # 64x112x112
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pad=1, ceil_mode=False, pool=P.Pooling.MAX)  # 64x56x56

        for num in xrange(len(stages)):  # num = 0, 1, 2, 3
            for i in xrange(stages[num]):
                if i == 0:
                    stage_string = match_string
                    bottom_string = ['n.pool1', 'n.resx{}_elewise'.format(str(sum(stages[:1]))),
                                     'n.resx{}_elewise'.format(str(sum(stages[:2]))),
                                     'n.resx{}_elewise'.format(str(sum(stages[:3])))][num]
                else:
                    stage_string = resnext_string
                    bottom_string = 'n.resx{}_elewise'.format(str(sum(stages[:num]) + i))
                print num, i
                exec (stage_string.replace('(bottom)', bottom_string).
                      replace('(base)', str(2 ** num * 64)).
                      replace('(n)', str(sum(stages[:num]) + i + 1)).
                      replace('(s)', str(int(num > 0) + 1)).
                      replace('(c)', str(card)))

        exec 'n.pool_ave = L.Pooling(n.resx{}_elewise, pool=P.Pooling.AVE, global_pooling=True)'.format(
            str(sum(stages)))
        n.classifier = L.InnerProduct(n.pool_ave, num_output=self.classifier_num,
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
