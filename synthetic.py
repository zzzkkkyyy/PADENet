import os, sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import math
import pdb
import glob

from torch.autograd import Variable

class SyntheticDetection(data.Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.image_root = ""
        self.anno_root = ""
        self.calib_root = ""
        self.mode = mode
        self.wfov = 360
        self.hfov = 52.7

        self.image_list = sorted(list(glob.glob(os.path.join(root, 'image', '*.png'))))
        self.anno_list = sorted(list(glob.glob(os.path.join(root, 'label', '*.txt'))))
        self.depth_list = sorted(list(glob.glob(os.path.join(root, 'depth', '*.png'))))
        self.index_list = list(np.arange(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_COLOR)

        image = cv2.resize(image, (2048, 320))
        image = image.astype(np.float32)
        image -= [104, 117, 123]
        image /= 128.
        image = image.transpose(2, 0, 1)

        return image
