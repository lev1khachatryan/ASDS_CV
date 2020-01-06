'''
Train or make inference the style transfer network
'''

from __future__ import print_function

import tensorflow as tf
from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = True

# for training
TRAINING_CONTENT_DIR = '../../_input/content'
TRAINING_STYLE_DIR = '../../_input/style'
ENCODER_WEIGHTS_PATH = '../../vgg19_normalised.npz'
LOGGING_PERIOD = 20

STYLE_WEIGHTS = [3.0]
CONTENT_WEIGHTS = [1.0]
MODEL_SAVE_PATHS = ['../../models/style_weight_2e0.ckpt']

# # for inferring (stylize)
# INFERRING_CONTENT_DIR = 'images/content'
# INFERRING_STYLE_DIR = 'images/style'
# OUTPUTS_DIR = 'images/output'


def main():

    if IS_TRAINING:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        tf.reset_default_graph()

        for style_weight, content_weight, model_save_path in zip(STYLE_WEIGHTS, CONTENT_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight))

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()