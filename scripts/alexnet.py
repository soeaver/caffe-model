import caffe
from caffe import layers as L
from caffe import params as P


def conv_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, group=group,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    relu = L.ReLU(conv, in_place=True)
    return conv, relu


def factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=1),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def fc_relu_drop(bottom, num_output=1024, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=num_output,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='xavier', std=1),
                        bias_filler=dict(type='constant', value=0.2))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def accuracy_top1_top5(bottom, label):
    accuracy_top1 = L.Accuracy(bottom, label, include=dict(phase=1))
    accuracy_top5 = L.Accuracy(bottom, label, include=dict(phase=1), accuracy_param=dict(top_k=5))
    return accuracy_top1, accuracy_top5


class AlexNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def alexnet_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1, n.relu1 = conv_relu(n.data, num_output=96, kernel_size=11, stride=4)  # 96x55x55
        n.norm1 = L.LRN(n.conv1, local_size=5, alpha=0.0001, beta=0.75)
        n.pool1 = L.Pooling(n.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 96x27x27

        n.conv2, n.relu2 = conv_relu(n.pool1, num_output=256, kernel_size=5, pad=2, group=2)  # 256x27x27
        n.norm2 = L.LRN(n.conv2, local_size=5, alpha=0.0001, beta=0.75)
        n.pool2 = L.Pooling(n.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x13x13

        n.conv3, n.relu3 = conv_relu(n.pool2, num_output=384, kernel_size=3, pad=1)  # 384x13x13
        n.conv4, n.relu4 = conv_relu(n.conv3, num_output=384, kernel_size=3, pad=1, group=2)  # 384x13x13

        n.conv5, n.relu5 = conv_relu(n.conv4, num_output=256, kernel_size=3, pad=1, group=2)  # 256x13x13
        n.pool5 = L.Pooling(n.conv5, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x6x16

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, num_output=4096)  # 4096x1x1
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, num_output=4096)  # 4096x1x1
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()

    def alexnet_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
            factorization_conv_bn_scale_relu(n.data, num_output=96, kernel_size=11, stride=4)  # 96x55x55
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 96x27x27

        n.conv2, n.conv2_bn, n.conv2_scale, n.conv2_relu = \
            factorization_conv_bn_scale_relu(n.pool1, num_output=256, kernel_size=5, pad=2)  # 256x27x27
        n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x13x13

        n.conv3, n.conv3_bn, n.conv3_scale, n.conv3_relu = \
            factorization_conv_bn_scale_relu(n.pool2, num_output=384, kernel_size=3, pad=1)  # 384x13x13

        n.conv4, n.conv4_bn, n.conv4_scale, n.conv4_relu = \
            factorization_conv_bn_scale_relu(n.conv3, num_output=384, kernel_size=3, pad=1)  # 384x13x13

        n.conv5, n.conv5_bn, n.conv5_scale, n.conv5_relu = \
            factorization_conv_bn_scale_relu(n.conv4, num_output=256, kernel_size=3, pad=1)  # 256x13x13
        n.pool5 = L.Pooling(n.conv5, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x6x16

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, num_output=4096)  # 4096x1x1
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, num_output=4096)  # 4096x1x1
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0))
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()
