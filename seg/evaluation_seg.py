import sys

sys.path.append('/home/prmct/workspace/PSPNet-0120/python/')

import caffe
import cv2
import numpy as np
import datetime

gpu_mode = True
gpu_id = 3
data_root = '/home/prmct/Database/VOC_PASCAL/VOC2012_test/JPEGImages/'
val_file = 'test_205.txt'
save_root = './predict205_40000_ms/'
model_weights = 'psp_resnext101_32x4d_coco_sbd_iter_40000.caffemodel'
model_deploy = 'deploy_psp_resnext101_32x4d_merge_bn_scale.prototxt'
prob_layer = 'prob'  # output layer, normally Softmax
class_num = 21
base_size = 512
crop_size = 473
raw_scale = 57.5  # image scale factor, 1.0 or 128.0
# mean_value = np.array([104.008, 116.669, 122.675])
# mean_value = np.array([128, 128, 128])
mean_value = np.array([103.52, 116.28, 123.675])
scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]   # multi scale
# scale_array = [1.0]  # single scale
flip = True
class_offset = 0
crf = False
crf_deploy = '/home/prmct/Program/segmentation/deploy_crf.prototxt'
crf_factor = 4.0

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)

if crf:
    net_crf = caffe.Net(crf_deploy, caffe.TEST)


def eval_batch():
    eval_images = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip())

    skip_num = 0
    eval_len = len(eval_images)
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(data_root + eval_images[i + skip_num] + '.jpg')
        h, w, d = _img.shape

        score_map = np.zeros((h, w, class_num), dtype=np.float32)
        for j in scale_array:
            long_size = float(base_size) * j + 1
            ratio = long_size / max(h, w)
            new_size = (int(w * ratio), int(h * ratio))
            _scale = cv2.resize(_img, new_size)
            score_map += cv2.resize(scale_process(_scale), (w, h))
        score_map /= len(scale_array)

        if crf:
            tmp_data = np.asarray([_img.transpose(2, 0, 1)], dtype=np.float32)
            tmp_score = np.asarray([score_map.transpose(2, 0, 1)], dtype=np.float32)
            net_crf.blobs['data'].reshape(*tmp_data.shape)
            net_crf.blobs['data'].data[...] = tmp_data / raw_scale
            net_crf.blobs['data_dim'].data[...] = [[[h, w]]]
            net_crf.blobs['score'].reshape(*tmp_score.shape)
            net_crf.blobs['score'].data[...] = tmp_score * crf_factor
            net_crf.forward()
            score_map = net_crf.blobs[prob_layer].data[0].transpose(1, 2, 0)

        cv2.imwrite(save_root + eval_images[i + skip_num] + '.png', score_map.argmax(2) + class_offset)
        print 'Testing image: ' + str(i + 1) + '/' + str(eval_len) + '  ' + str(eval_images[i + skip_num])
    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    print '\n{} images has been tested. \nThe model is: {}'.format(str(eval_len), model_weights)


def scale_process(_scale):
    sh, sw, sd = _scale.shape
    _scale = np.asarray(_scale, dtype=np.float32)
    long_size = max(sh, sw)
    short_size = min(sh, sw)
    if long_size <= crop_size:
        input_data = pad_img(_scale - mean_value)
        score = caffe_process(input_data)[:sh, :sw, :]
    else:
        stride_rate = 2.0 / 3
        stride = np.ceil(crop_size * stride_rate)
        _pad = _scale
        if short_size < crop_size:
            _pad = pad_img(_scale - mean_value) + mean_value

        ph, pw, pd = _pad.shape
        h_grid = int(np.ceil(float(ph - crop_size) / stride)) + 1
        w_grid = int(np.ceil(float(pw - crop_size) / stride)) + 1
        data_scale = np.zeros((ph, pw, class_num), dtype=np.float32)
        count_scale = np.zeros((ph, pw, class_num), dtype=np.float32)
        for grid_yidx in xrange(0, h_grid):
            for grid_xidx in xrange(0, w_grid):
                s_x = int(grid_xidx * stride)
                s_y = int(grid_yidx * stride)
                e_x = min(s_x + crop_size, pw)
                e_y = min(s_y + crop_size, ph)
                s_x = int(e_x - crop_size)
                s_y = int(e_y - crop_size)
                _sub = _pad[s_y:e_y, s_x:e_x, :]
                count_scale[s_y:e_y, s_x:e_x, :] += 1.0
                input_data = pad_img(_sub - mean_value)
                data_scale[s_y:e_y, s_x:e_x, :] += caffe_process(input_data)
        score = data_scale / count_scale
        score = score[:sh, :sw, :]

    return score


def pad_img(_scale):
    sh, sw, sd = _scale.shape
    if sh < crop_size:
        _pad = np.zeros((crop_size, sw, sd), dtype=np.float32)
        _pad[:sh, :, :] = _scale
        _scale = _pad
    sh, sw, sd = _scale.shape
    if sw < crop_size:
        _pad = np.zeros((sh, crop_size, sd), dtype=np.float32)
        _pad[:, :sw, :] = _scale
        _scale = _pad

    return _scale


def caffe_process(_input):
    h, w, d = _input.shape
    _score = np.zeros((h, w, class_num), dtype=np.float32)
    if flip:
        _flip = _input[:, ::-1]
        _flip = _flip.transpose(2, 0, 1)
        _flip = _flip.reshape((1,) + _flip.shape)
        net.blobs['data'].reshape(*_flip.shape)
        net.blobs['data'].data[...] = _flip / raw_scale
        # net.blobs['data_dim'].data[...] = [[[h, w]]]
        net.forward()
        _score += net.blobs[prob_layer].data[0].transpose(1, 2, 0)[:, ::-1]

    _input = _input.transpose(2, 0, 1)
    _input = _input.reshape((1,) + _input.shape)
    net.blobs['data'].reshape(*_input.shape)
    net.blobs['data'].data[...] = _input / raw_scale
    # net.blobs['data_dim'].data[...] = [[[h, w]]]
    net.forward()
    _score += net.blobs[prob_layer].data[0].transpose(1, 2, 0)

    return _score / int(flip + 1)

if __name__ == '__main__':
    eval_batch()

