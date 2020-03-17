import os
import glob
from PIL import Image
import random
import numpy as np

class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
    			 height_of_image, width_of_image, num_channels, num_classes):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.*"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.*"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.*"), recursive=True)

#         random.shuffle(self.train_paths)
#         random.shuffle(self.val_paths)
#         random.shuffle(self.test_paths)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.height_of_image = height_of_image
        self.width_of_image = width_of_image
        self.num_channels = num_channels
        self.num_classes = num_classes

    def load_image(self, path, is_flattened = False):
        im = np.asarray(Image.open(path).resize((self.width_of_image, self.height_of_image)))
        if path.rsplit('\\', 2)[-2] == 'london':
        	lbl = [1., 0.]
        elif path.rsplit('\\', 2)[-2] == 'yerevan':
        	lbl = [0., 1.]

        if is_flattened:
            im = im.reshape(self.height_of_image * self.width_of_image * self.num_channels)

        return im, lbl

    def batch_data_loader(self, batch_size, file_paths, index, is_flattened = False, randomization = False):
        ims = []
        lbls = []
        
        if index == 0 or randomization:
            random.shuffle(file_paths)
        
        while batch_size >= 1 and (len(file_paths) - index > 0):
            im, lbl = self.load_image(file_paths[index], is_flattened)
            ims.append(im)
            lbls.append(lbl)
            batch_size -= 1
            index += 1
        imgs = ims
        # imgs = np.array(ims)
        # imgs = imgs.reshape(-1, self.height_of_image, self.width_of_image, self.num_channels)
        # imgs = imgs - imgs.mean() / imgs.std()

        return imgs, np.array(lbls)

    
    def train_data_loader(self, index, randomization = False):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index, randomization = randomization)

    def val_data_loader(self, index, randomization = False):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index, randomization = randomization)
    
    
    def get_train_data_size(self):
        return len(self.train_paths)
    
    def get_val_data_size(self):
        return len(self.val_paths)
    
    def get_test_data_size(self):
        return len(self.test_paths)
    
    
    def all_train_data_loader(self, is_flattened = False):
        return self.batch_data_loader(self.get_train_data_size(), self.train_paths, 0)
    
    def all_val_data_loader(self, is_flattened = False):
        return self.batch_data_loader(self.get_val_data_size(), self.val_paths, 0)
    
    def all_test_data_loader(self, is_flattened = False):
        ims = []
        for index in range(len(self.test_paths)):
            im = np.asarray(Image.open(self.test_paths[index]).resize((self.width_of_image, self.height_of_image)))
            if is_flattened:
                im = im.reshape(self.height_of_image * self.width_of_image * self.num_channels)
            ims.append(im)

        return ims