from caffe import layers as L
from caffe import params as P
import caffe


def conv_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    relu = L.ReLU(conv, in_place=True)
    return conv, relu


def fc_relu_drop(bottom, fc_num_output=4096, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=fc_num_output,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian', std=0.01),
                        bias_filler=dict(type='constant', value=0)
                        )
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True, dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)

    return conv, bn, scale, relu


def accuracy_top1_top5(bottom, label):
    accuracy_top1 = L.Accuracy(bottom, label, include=dict(phase=1))
    accuracy_top5 = L.Accuracy(bottom, label, include=dict(phase=1), accuracy_param=dict(top_k=5))
    return accuracy_top1, accuracy_top5


class NIN(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def nin_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1, n.relu0 = conv_relu(n.data, num_output=96, kernel_size=11, stride=4)  # 96x53x53
        n.cccp1, n.relu1 = conv_relu(n.conv1, num_output=96, kernel_size=1, stride=1)
        n.cccp2, n.relu2 = conv_relu(n.cccp1, num_output=96, kernel_size=1, stride=1)
        n.pool1 = L.Pooling(n.cccp2, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 96x26x26

        n.conv2, n.relu3 = conv_relu(n.pool1, num_output=256, kernel_size=5, stride=1, pad=2)  # 256x26x26
        n.cccp3, n.relu4 = conv_relu(n.conv2, num_output=256, kernel_size=1, stride=1)
        n.cccp4, n.relu5 = conv_relu(n.cccp3, num_output=256, kernel_size=1, stride=1)
        n.pool2 = L.Pooling(n.cccp4, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 256x13x13

        n.conv3, n.relu6 = conv_relu(n.pool2, num_output=384, kernel_size=3, stride=1, pad=1)  # 384x13x13
        n.cccp5, n.relu7 = conv_relu(n.conv3, num_output=384, kernel_size=1, stride=1)
        n.cccp6, n.relu8 = conv_relu(n.cccp5, num_output=384, kernel_size=1, stride=1)
        n.pool3 = L.Pooling(n.cccp6, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 384x6x6
        n.drop7 = L.Dropout(n.pool3, in_place=True, dropout_param=dict(dropout_ratio=0.5))

        n.conv4, n.relu9 = conv_relu(n.pool3, num_output=1024, kernel_size=3, stride=1, pad=1)  # 1024x6x6
        n.cccp7, n.relu10 = conv_relu(n.conv4, num_output=1024, kernel_size=1, stride=1)
        n.cccp8, n.relu11 = conv_relu(n.cccp7, num_output=1024, kernel_size=1, stride=1)
        n.pool4 = L.Pooling(n.cccp8, pool=P.Pooling.AVE, kernel_size=6, stride=1)  # 1024x1x1
        n.classifier = L.InnerProduct(n.pool4, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='gaussian', std=0.01),
                                      bias_filler=dict(type='constant', value=0)
                                      )

        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.pool4, n.label)

        return n.to_proto()

    def nin_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1, n.bn0, n.scale0, n.relu0 = conv_bn_scale_relu(n.data, num_output=96, kernel_size=11, stride=4)
        n.cccp1, n.bn1, n.scale1, n.relu1 = conv_bn_scale_relu(n.conv1, num_output=96, kernel_size=1, stride=1)
        n.cccp2, n.bn2, n.scale2, n.relu2 = conv_bn_scale_relu(n.cccp1, num_output=96, kernel_size=1, stride=1)
        n.pool1 = L.Pooling(n.cccp2, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 96x26x26

        n.conv2, n.bn3, n.scale3, n.relu3 = conv_bn_scale_relu(n.pool1, num_output=256, kernel_size=5, stride=1, pad=2)
        n.cccp3, n.bn4, n.scale4, n.relu4 = conv_bn_scale_relu(n.conv2, num_output=256, kernel_size=1, stride=1)
        n.cccp4, n.bn5, n.scale5, n.relu5 = conv_bn_scale_relu(n.cccp3, num_output=256, kernel_size=1, stride=1)
        n.pool2 = L.Pooling(n.cccp4, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 256x13x13

        n.conv3, n.bn6, n.scale6, n.relu6 = conv_bn_scale_relu(n.pool2, num_output=384, kernel_size=3, stride=1, pad=1)
        n.cccp5, n.bn7, n.scale7, n.relu7 = conv_bn_scale_relu(n.conv3, num_output=384, kernel_size=1, stride=1)
        n.cccp6, n.bn8, n.scale8, n.relu8 = conv_bn_scale_relu(n.cccp5, num_output=384, kernel_size=1, stride=1)
        n.pool3 = L.Pooling(n.cccp6, pool=P.Pooling.MAX, kernel_size=3, stride=2)  # 384x6x6
        n.drop7 = L.Dropout(n.pool3, in_place=True, dropout_param=dict(dropout_ratio=0.5))

        n.conv4, n.bn9, n.scale9, n.relu9 = conv_bn_scale_relu(n.pool3, num_output=1024, kernel_size=3, stride=1, pad=1)
        n.cccp7, n.bn10, n.scale10, n.relu10 = conv_bn_scale_relu(n.conv4, num_output=1024, kernel_size=1, stride=1)
        n.cccp8, n.bn11, n.scale11, n.relu11 = conv_bn_scale_relu(n.cccp7, num_output=1024, kernel_size=1, stride=1)
        n.pool4 = L.Pooling(n.cccp8, pool=P.Pooling.AVE, kernel_size=6, stride=1)  # 1024x1x1
        n.classifier = L.InnerProduct(n.pool4, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='gaussian', std=0.01),
                                      bias_filler=dict(type='constant', value=0)
                                      )

        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.pool4, n.label)

        return n.to_proto()
