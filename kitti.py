import os, sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import math
import pdb

from torch.autograd import Variable

train_range = (0, 7200)
valid_range = (7200, 7481)

class KittiDetection(data.Dataset):
    def __init__(self, root, mode='train', augment=None):
        self.root = root
        self.augment = augment
        self.image_root = ""
        self.anno_root = ""
        self.calib_root = ""
        self.mode = mode
        self.wfov = 82.5
        self.hfov = 29.7

        if mode == 'train':
            self.image_left_root = os.path.join(root, 'data_object_eqimage_2/training/image_2')
            self.image_right_root = os.path.join(root, 'data_object_eqimage_2/training/image_3')
            self.anno_root = os.path.join(root, 'data_object_eqlabel_2/training/label_2')
            self.calib_root = os.path.join(root, 'data_object_calib/training/calib')

            self.id_list = ["{:0>6}".format(i) for i in range(train_range[0], train_range[1])]
            self.image_list = sorted([item.rstrip('\n') + '.png' for item in self.id_list])
            self.anno_list = sorted([item.rstrip('\n') + '.txt' for item in self.id_list])
            self.calib_list = sorted([item.rstrip('\n') + '.txt' for item in self.id_list])
            self.index_list = list(np.arange(len(self.id_list)))

        if mode == 'val':
            self.image_left_root = os.path.join(root, 'data_object_eqimage_2/training/image_2')
            self.image_right_root = os.path.join(root, 'data_object_eqimage_2/training/image_3')
            self.anno_root = os.path.join(root, 'data_object_eqlabel_2/training/label_2')
            self.calib_root = os.path.join(root, 'data_object_calib/training/calib')

            self.id_list = ["{:0>6}".format(i) for i in range(valid_range[0], valid_range[1])]
            self.image_list = sorted([item.rstrip('\n') + '.png' for item in self.id_list])
            self.anno_list = sorted([item.rstrip('\n') + '.txt' for item in self.id_list])
            self.calib_list = sorted([item.rstrip('\n') + '.txt' for item in self.id_list])
            self.index_list = list(np.arange(len(self.id_list)))


    def __len__(self):
        return len(self.image_list)

    def get_filename(self, fileindex):
        return self.id_list[int(fileindex)]

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_left_root, self.image_list[index]), cv2.IMREAD_COLOR)
        image_pair = cv2.imread(os.path.join(self.image_right_root, self.image_list[index]), cv2.IMREAD_COLOR)

        image = cv2.resize(image, (960, 320))
        image = image.astype(np.float32)
        image -= [104, 117, 123]
        image /= 128.
        image = image.transpose(2, 0, 1)

        image_pair = cv2.resize(image_pair, (960, 320))
        image_pair = image_pair.astype(np.float32)
        image_pair -= [104, 117, 123]
        image_pair /= 128.
        image_pair = image_pair.transpose(2, 0, 1)

        return image, image_pair
