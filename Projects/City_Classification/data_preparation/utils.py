# Used functions for data preparation
from __future__ import print_function

import sys
import warnings
import traceback
import numpy as np
import shutil
import os
from os import remove
from os import listdir
import os.path
from os.path import join
from datetime import datetime
from scipy.misc import imread, imresize
from PIL import Image

def create_dir(dir_name, relative_path):
    """
    Create new directory if not exists
    --------------
    Parameters:
        dir_name (string)      - name of directory we want to create
        relative_path (string) - absolute path of directory we want to create
    Returns:
        path (string)          - full path of directory
    --------------
    """
    
    path = relative_path + dir_name
    if not os.path.exists(path):
        os.mkdir(path)
    return path + '/'

def list_train_images(directory):
    """
    get all images in specified directory
    --------------
    Parameters:
        directory (string) - absolute path of directory from which we want to take images
    Returns:
        list (all images)
    --------------
    """
    subFolderList = []
    for x in os.listdir(directory):
        if os.path.isdir(directory + '\\' + x):
            if x != 'test':
                subFolderList.append(x)
                
    all_files = []
    for x in subFolderList:
        all_files += [directory  + x +'\\' + y for y in os.listdir(directory + x)]
    return all_files

def list_test_images(directory):
    """
    get all images in specified directory
    --------------
    Parameters:
        directory (string) - absolute path of directory from which we want to take images
    Returns:
        list (all images)
    --------------
    """
    images = []
    for file in listdir(directory):
        images.append(join(directory, file))
    return images

def discard_bad_images(dir_path, where_to_save_names, paths): 
    warnings.filterwarnings('error')
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # paths = list_images(dir_path)

    print('\nOrigin files number: %d\n' % len(paths))

    num_delete = 0
    path_delete = []

    for path in paths:
        is_continue = False

        try:
            image = Image.open(path)
            image.verify()
            image = imread(path)
            # image = imread(path, mode='RGB')
        except Warning as warn:
            is_continue = True
            num_delete += 1
            path_delete.append(path)
            remove(path)

            print('>>> Warning happens! Removes image <%s>' % path)
            print('Warning detail:\n%s\n' % str(warn))
        except Exception as exc:
            is_continue = True
            num_delete += 1
            path_delete.append(path)
            remove(path)

            print('>>> Exception happens! Removes image <%s>' % path)
            print('Exception detail:\n%s\n' % str(exc))

        if is_continue:
            continue

        if len(image.shape) != 3 or image.shape[2] != 3:
            num_delete += 1
            path_delete.append(path)
            remove(path)

            print('>>> Found an image with shape: %s; Now removes it: <%s>\n' % (str(image.shape), path))
        else:
            height, width, _ = image.shape

            if height < width:
                new_height = 512
                new_width  = int(width * new_height / height)
            else:
                new_width  = 512
                new_height = int(height * new_width / width)

            try:
                image = imresize(image, [new_height, new_width], interp='nearest')
            except:
                num_delete += 1
                path_delete.append(path)
                remove(path)
                
                print('>>> Fails to resize an image! Now removes it: <%s>\n' % path)
                traceback.print_exception(*sys.exc_info())

    print('\n>>>>> delete %d files! Current number of files: %d\n' % (num_delete, len(paths) - num_delete))

    with open(os.path.join(where_to_save_names, 'corrupted_images.txt'), 'a+') as f:
        for item in path_delete:
            f.write("%s\n" % item)
