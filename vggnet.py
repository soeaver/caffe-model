from caffe import layers as L
from caffe import params as P
import caffe


def conv_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=1):
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


def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=1):
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


class VggNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def vgg_16_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.relu1_1 = conv_relu(n.data, num_output=64)
        n.conv1_2, n.relu1_2 = conv_relu(n.conv1_1, num_output=64)
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)  # 64x112x112

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, num_output=128)
        n.conv2_2, n.relu2_2 = conv_relu(n.conv2_1, num_output=128)
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)  # 128x56x56

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, num_output=256)
        n.conv3_2, n.relu3_2 = conv_relu(n.conv3_1, num_output=256)
        n.conv3_3, n.relu3_3 = conv_relu(n.conv3_2, num_output=256)
        n.pool3 = L.Pooling(n.conv3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)  # 256x28x28

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, num_output=512)
        n.conv4_2, n.relu4_2 = conv_relu(n.conv4_1, num_output=512)
        n.conv4_3, n.relu4_3 = conv_relu(n.conv4_2, num_output=512)
        n.pool4 = L.Pooling(n.conv4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)  # 512x14x14

        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, num_output=512)
        n.conv5_2, n.relu5_2 = conv_relu(n.conv5_1, num_output=512)
        n.conv5_3, n.relu5_3 = conv_relu(n.conv5_2, num_output=512)
        n.pool5 = L.Pooling(n.conv5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)  # 512x7x7

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, fc_num_output=4096, dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, fc_num_output=4096, dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0)
                               )
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()

    def vgg_16_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1_1, n.bn1_1, n.scale1_1, n.relu1_1 = conv_bn_scale_relu(n.data, num_output=64)
        n.conv1_2, n.bn1_2, n.scale1_2, n.relu1_2 = conv_bn_scale_relu(n.conv1_1, num_output=64)
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.bn2_1, n.scale2_1, n.relu2_1 = conv_bn_scale_relu(n.pool1, num_output=128)
        n.conv2_2, n.bn2_2, n.scale2_2, n.relu2_2 = conv_bn_scale_relu(n.conv2_1, num_output=128)
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.bn3_1, n.scale3_1, n.relu3_1 = conv_bn_scale_relu(n.pool2, num_output=256)
        n.conv3_2, n.bn3_2, n.scale3_2, n.relu3_2 = conv_bn_scale_relu(n.conv3_1, num_output=256)
        n.conv3_3, n.bn3_3, n.scale3_3, n.relu3_3 = conv_bn_scale_relu(n.conv3_2, num_output=256)
        n.pool3 = L.Pooling(n.conv3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv4_1, n.bn4_1, n.scale4_1, n.relu4_1 = conv_bn_scale_relu(n.pool3, num_output=512)
        n.conv4_2, n.bn4_2, n.scale4_2, n.relu4_2 = conv_bn_scale_relu(n.conv4_1, num_output=512)
        n.conv4_3, n.bn4_3, n.scale4_3, n.relu4_3 = conv_bn_scale_relu(n.conv4_2, num_output=512)
        n.pool4 = L.Pooling(n.conv4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv5_1, n.bn5_1, n.scale5_1, n.relu5_1 = conv_bn_scale_relu(n.pool4, num_output=512)
        n.conv5_2, n.bn5_2, n.scale5_2, n.relu5_2 = conv_bn_scale_relu(n.conv5_1, num_output=512)
        n.conv5_3, n.bn5_3, n.scale5_3, n.relu5_3 = conv_bn_scale_relu(n.conv5_2, num_output=512)
        n.pool5 = L.Pooling(n.conv5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, fc_num_output=4096, dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, fc_num_output=4096, dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0)
                               )
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()

    def vgg_19_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))
        n.conv1_1, n.relu1_1 = conv_relu(n.data, num_output=64)
        n.conv1_2, n.relu1_2 = conv_relu(n.conv1_1, num_output=64)
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, num_output=128)
        n.conv2_2, n.relu2_2 = conv_relu(n.conv2_1, num_output=128)
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, num_output=256)
        n.conv3_2, n.relu3_2 = conv_relu(n.conv3_1, num_output=256)
        n.conv3_3, n.relu3_3 = conv_relu(n.conv3_2, num_output=256)
        n.conv3_4, n.relu3_4 = conv_relu(n.conv3_3, num_output=256)
        n.pool3 = L.Pooling(n.conv3_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, num_output=512)
        n.conv4_2, n.relu4_2 = conv_relu(n.conv4_1, num_output=512)
        n.conv4_3, n.relu4_3 = conv_relu(n.conv4_2, num_output=512)
        n.conv4_4, n.relu4_4 = conv_relu(n.conv4_3, num_output=512)
        n.pool4 = L.Pooling(n.conv4_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, num_output=512)
        n.conv5_2, n.relu5_2 = conv_relu(n.conv5_1, num_output=512)
        n.conv5_3, n.relu5_3 = conv_relu(n.conv5_2, num_output=512)
        n.conv5_4, n.relu5_4 = conv_relu(n.conv5_3, num_output=512)
        n.pool5 = L.Pooling(n.conv5_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, fc_num_output=4096, dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, fc_num_output=4096, dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0)
                               )
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()

    def vgg_19_bn_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=mirror))

        n.conv1_1, n.bn1_1, n.scale1_1, n.relu1_1 = conv_bn_scale_relu(n.data, num_output=64)
        n.conv1_2, n.bn1_2, n.scale1_2, n.relu1_2 = conv_bn_scale_relu(n.conv1_1, num_output=64)
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.bn2_1, n.scale2_1, n.relu2_1 = conv_bn_scale_relu(n.pool1, num_output=128)
        n.conv2_2, n.bn2_2, n.scale2_2, n.relu2_2 = conv_bn_scale_relu(n.conv2_1, num_output=128)
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.bn3_1, n.scale3_1, n.relu3_1 = conv_bn_scale_relu(n.pool2, num_output=256)
        n.conv3_2, n.bn3_2, n.scale3_2, n.relu3_2 = conv_bn_scale_relu(n.conv3_1, num_output=256)
        n.conv3_3, n.bn3_3, n.scale3_3, n.relu3_3 = conv_bn_scale_relu(n.conv3_2, num_output=256)
        n.conv3_4, n.bn3_4, n.scale3_4, n.relu3_4 = conv_bn_scale_relu(n.conv3_3, num_output=256)
        n.pool3 = L.Pooling(n.conv3_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv4_1, n.bn4_1, n.scale4_1, n.relu4_1 = conv_bn_scale_relu(n.pool3, num_output=512)
        n.conv4_2, n.bn4_2, n.scale4_2, n.relu4_2 = conv_bn_scale_relu(n.conv4_1, num_output=512)
        n.conv4_3, n.bn4_3, n.scale4_3, n.relu4_3 = conv_bn_scale_relu(n.conv4_2, num_output=512)
        n.conv4_4, n.bn4_4, n.scale4_4, n.relu4_4 = conv_bn_scale_relu(n.conv4_3, num_output=512)
        n.pool4 = L.Pooling(n.conv4_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv5_1, n.bn5_1, n.scale5_1, n.relu5_1 = conv_bn_scale_relu(n.pool4, num_output=512)
        n.conv5_2, n.bn5_2, n.scale5_2, n.relu5_2 = conv_bn_scale_relu(n.conv5_1, num_output=512)
        n.conv5_3, n.bn5_3, n.scale5_3, n.relu5_3 = conv_bn_scale_relu(n.conv5_2, num_output=512)
        n.conv5_4, n.bn5_4, n.scale5_4, n.relu5_4 = conv_bn_scale_relu(n.conv5_3, num_output=512)
        n.pool5 = L.Pooling(n.conv5_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.fc6, n.relu6, n.drop6 = fc_relu_drop(n.pool5, fc_num_output=4096, dropout_ratio=0.5)
        n.fc7, n.relu7, n.drop7 = fc_relu_drop(n.fc6, fc_num_output=4096, dropout_ratio=0.5)
        n.fc8 = L.InnerProduct(n.fc7, num_output=self.classifier_num,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type='gaussian', std=0.01),
                               bias_filler=dict(type='constant', value=0)
                               )
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1, n.accuracy_top5 = accuracy_top1_top5(n.fc8, n.label)

        return n.to_proto()
