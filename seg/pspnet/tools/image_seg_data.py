import cv2
import sys

sys.path.append('/home/prmct/workspace/py-RFCN-priv/caffe-priv/python')

import caffe

import numpy as np


class ImageSegDataLayer(caffe.Layer):
    def get_gpu_id(self, gpu_id=0):
        self.gpu_id = gpu_id
        if self.shuffle:
            np.random.seed(self.gpu_id)
            np.random.shuffle(self.indices)

    def setup(self, bottom, top):
        print self.param_str
        params = eval(self.param_str)

        self.color_factor = np.array(params.get('color_factor', (0.95, 1.05)))  #  (0.95, 1.05)
        self.contrast_factor = np.array(params.get('contrast_factor',  (0.95, 1.05)))  #  (0.95, 1.05)
        self.brightness_factor = np.array(params.get('brightness_factor',  (0.95, 1.05)))  #  (0.95, 1.05)
        self.mirror = params.get('mirror', True)
        self.gaussian_blur = params.get('gaussian_blur', True)
        self.scale_factor = np.array(params.get('scale_factor', (0.75, 2.0)))  # (0.75, 2.0)
        self.rotation_factor = np.array(params.get('rotation_factor', (-10, 10)))  # (-10, 10)

        self.crop_size = int(params.get('crop_size', 513))
        self.ignore_label = int(params.get('ignore_label', 255))
        self.mean = np.array(params.get('mean', (102.98, 115.947, 122.772)), dtype=np.float32)
        self.scale = float(params.get('scale', 1.0))

        self.root_dir = params['root_dir']
        self.source = params['source']
        self.batch_size = int(params.get('batch_size', 1))
        self.shuffle = params.get('shuffle', True)

        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")  # data layers have no bottoms
        if len(self.color_factor) != 2:
            raise Exception("'color_factor' must have 2 values for factor range.")
        if len(self.contrast_factor) != 2:
            raise Exception("'contrast_factor' must have 2 values for factor range.")
        if len(self.brightness_factor) != 2:
            raise Exception("'brightness_factor' must have 2 values for factor range.")
        if len(self.mean) != 3:
            raise Exception("'mean' must have 3 values for B G R.")
        if len(self.scale_factor) != 2:
            raise Exception("'scale_factor' must have 2 values for factor range.")
        if self.crop_size <= 0:
            raise Exception("'Need positive crop_size.")

        self.indices = open(self.source, 'r').read().splitlines()
        self.epoch_num = len(self.indices)
        self.idx = 0

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)  # for images
        top[1].reshape(self.batch_size, 1, self.crop_size, self.crop_size)  # for labels

    def forward(self, bottom, top):
        batch_img = []
        batch_label = []
        for _ in xrange(self.batch_size):
            _img = cv2.imread('{}{}'.format(self.root_dir, self.indices[self.idx].split(' ')[0]))
            _label = cv2.imread('{}{}'.format(self.root_dir, self.indices[self.idx].split(' ')[1]), 0)

            if _img.shape[:2] != _label.shape:
                raise Exception("Need to define two tops: data and label.")

            aug_img, aug_label = self.augmentation(_img, _label)
            batch_img.append(aug_img.transpose((2, 0, 1)))
            batch_label.append([aug_label])

            self.idx += 1
            if self.idx == self.epoch_num:
                self.idx = 0
                if self.shuffle:
                    np.random.seed(self.gpu_id)
                    np.random.shuffle(self.indices)
        batch_img = np.asarray(batch_img)
        batch_label = np.asarray(batch_label)

        top[0].data[...] = batch_img
        top[1].data[...] = batch_label

    def backward(self, top, propagate_down, bottom):
        pass

    def augmentation(self, img, label):
        ori_h, ori_w = img.shape[:2]

        _color = 1.0
        _contrast = 1.0
        _brightness = 1.0

        if self.color_factor[0] != 0 and self.color_factor[1] != 0 and self.color_factor[0] < self.color_factor[1]:
            _color = np.random.randint(int(self.color_factor[0] * 100),
                                       int(self.color_factor[1] * 100)) / 100.0

        if self.contrast_factor[0] != 0 and self.contrast_factor[1] != 0 and self.contrast_factor[0] < \
                self.contrast_factor[1]:
            _contrast = np.random.randint(int(self.contrast_factor[0] * 100),
                                          int(self.contrast_factor[1] * 100)) / 100.0

        if self.brightness_factor[0] != 0 and self.brightness_factor[1] != 0 and self.brightness_factor[0] < \
                self.brightness_factor[1]:
            _brightness = np.random.randint(int(self.brightness_factor[0] * 100),
                                            int(self.brightness_factor[1] * 100)) / 100.0

        _HSV = np.dot(cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape((-1, 3)), 
                      np.array([[_color, 0, 0], [0, _contrast, 0], [0, 0, _brightness]]))
        _HSV_H = np.where(_HSV < 255, _HSV, 255)  
        img = cv2.cvtColor(np.uint8(_HSV_H.reshape((-1, img.shape[1], 3))), cv2.COLOR_HSV2BGR)

        if self.gaussian_blur:
            if not np.random.randint(0, 4):
                img = cv2.GaussianBlur(img, (3, 3), 0)

        img = np.asarray(img, dtype=np.float32)
        label = np.asarray(label, dtype=np.uint8)

        if self.mirror:
            if np.random.randint(0, 2):
                img = img[:, :: -1]
                label = label[:, :: -1]

        if self.scale_factor[0] != 0 and self.scale_factor[1] != 0 and self.scale_factor[0] < self.scale_factor[1]:
            _scale = np.random.randint(int(self.scale_factor[0] * 100),
                                       int(self.scale_factor[1] * 100)) / 100.0
            res_w = int(_scale * ori_w)
            res_h = int(_scale * ori_h)
            img = cv2.resize(img, (res_w, res_h))
            label = cv2.resize(label, (res_w, res_h), interpolation=cv2.cv.CV_INTER_NN)

        if self.rotation_factor[0] != 0 and self.rotation_factor[1] != 0 and self.rotation_factor[0] < \
                self.rotation_factor[1]:
            if np.random.randint(0, 2):
                _rotation = np.random.randint(int(self.rotation_factor[0] * 100),
                                              int(self.rotation_factor[1] * 100)) / 100.0
                tmp_h, tmp_w = img.shape[:2]
                rotate_mat = cv2.getRotationMatrix2D((tmp_w / 2, tmp_h / 2), _rotation,
                                                     1)
                img = cv2.warpAffine(img, rotate_mat, (tmp_w, tmp_h),
                                     borderValue=cv2.cv.Scalar(self.mean[0], self.mean[1], self.mean[2]))
                label = cv2.warpAffine(label, rotate_mat, (tmp_w, tmp_h), flags=cv2.cv.CV_INTER_NN,
                                       borderValue=cv2.cv.Scalar(self.ignore_label))

        # perform random crop
        pad_h = max(self.crop_size - img.shape[0], 0)
        pad_w = max(self.crop_size - img.shape[1], 0)
        pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=cv2.cv.Scalar(self.mean[0], self.mean[1], self.mean[2]))
        pad_label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                       value=cv2.cv.Scalar(self.ignore_label))
        off_h = np.random.randint(0, pad_img.shape[0] - self.crop_size + 1)
        off_w = np.random.randint(0, pad_img.shape[1] - self.crop_size + 1)
        aug_img = pad_img[off_h:off_h + self.crop_size, off_w:off_w + self.crop_size, :]
        aug_label = pad_label[off_h:off_h + self.crop_size, off_w:off_w + self.crop_size]

        # perform (x-mean)*scale
        aug_img -= self.mean
        aug_img *= self.scale

        return aug_img, aug_label
