import os
from glob import glob
import numpy as np
from torch.utils import data
from torchvision.transforms import Compose, ToTensor

import cv2
import os

import matplotlib.pyplot as plt

input_transform = Compose([
   ToTensor(),
])

class seg_data(data.Dataset):
    def __init__(self, root, image_height=384, image_width=288, mode = None):
        self.img_files = glob(os.path.join(root, 'origs', '*'))
        self.label_root = os.path.join(root, 'contours/')
        self.index = 0
        self.img_height = image_height
        self.img_width = image_width
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        path = self.img_files[index]
        img_name = os.path.basename(path)
        dirname = os.path.dirname(path)

        contour_path = os.path.join(dirname.replace('origs', 'contours'),
                                  '.'.join(img_name.split('.')[:-1]) + '.png')

        img = cv2.imread(path)[:,:,::-1]*1.0
        contour = cv2.imread(contour_path)[:,:,0]
        contour = np.where(contour>127, np.ones_like(contour)*255., np.zeros_like(contour)*1.0)


        # preproccess resize
        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        contour_resized = cv2.resize(contour, (self.img_width, self.img_height),
                                  interpolation=cv2.INTER_NEAREST)

        img_o = input_transform(img_resized) / 255.
        imgs = [img_o]

        contour_o = input_transform(np.expand_dims(contour_resized, axis = -1)) / 255.
        contours = [contour_o]

        return {'images':imgs, 'contours':contours}


class seg_data_test(data.Dataset):
    def __init__(self, root, image_height=384, image_width=288):
        self.img_files = glob(os.path.join(root, 'origs', '*'))
        self.index = 0
        self.img_height = image_height
        self.img_width = image_width

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        path = self.img_files[index]
        img = cv2.imread(path)[:, :, ::-1]
        name = '.'.join(os.path.basename(path).split('.')[:-1])

        # preproccess resize
        img_resized = cv2.resize(img, (self.img_width, self.img_height))*1.0
        contour_resized = np.zeros(shape = (self.img_height, self.img_width, 1)) * 1.0


        img_o = input_transform(img_resized) / 255.
        imgs = [img_o]

        contour_o = input_transform(contour_resized) / 255.
        contours = [contour_o]

        return {'images' : imgs, 'contours': contours, 'name': name}
