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
        self.label_root = os.path.join(root, 'masks/')
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

        mask_path = os.path.join(dirname.replace('origs', 'masks'),
                                  '.'.join(img_name.split('.')[:-1]) + '.png')

        img = cv2.imread(path)[:,:,::-1]*1.0
        mask = cv2.imread(mask_path)[:,:,0]
        mask = np.where(mask>127, np.ones_like(mask)*255., np.zeros_like(mask)*1.0)


        # preproccess resize
        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        mask_resized = cv2.resize(mask, (self.img_width, self.img_height),
                                  interpolation=cv2.INTER_NEAREST)

        if self.mode != 'test':
            # TODO make some real-time augmentations
            img_resized = img_resized
            mask_resized = mask_resized


        img_o = input_transform(img_resized) / 255.
        imgs = [img_o]

        mask_o = input_transform(np.expand_dims(mask_resized, axis = -1)) / 255.
        masks = [mask_o]

        return {'images':imgs, 'masks':masks}


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
        mask_resized = np.zeros(shape = (self.img_height, self.img_width, 1)) * 1.0


        img_o = input_transform(img_resized) / 255.
        imgs = [img_o]

        mask_o = input_transform(mask_resized) / 255.
        masks = [mask_o]

        return {'images' : imgs, 'masks': masks, 'name': name}
