'''
Train or make inference the style transfer network
'''

from __future__ import print_function

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf

import tensorflow as tf
from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = True

# for training
TRAINING_CONTENT_DIR = '../../_inference/content'
TRAINING_STYLE_DIR = '../../_inference/style'
ENCODER_WEIGHTS_PATH = '../../vgg19_normalised.npz'
LOGGING_PERIOD = 20

STYLE_WEIGHTS = [3.0]
CONTENT_WEIGHTS = [1.0]
LAMBDA1 = [1.0]
LAMBDA2 = [50.0]
MODEL_SAVE_PATHS = ['../../models/stylize_net.ckpt']

# for inferring (stylize)
INFERRING_CONTENT_DIR = '../../_inference/content'
INFERRING_STYLE_DIR = '../../_inference/style'
OUTPUTS_DIR = '../../_inference/output'


def main():

    if IS_TRAINING:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        tf.reset_default_graph()

        for style_weight, content_weight, lambda1, lambda2, model_save_path in zip(STYLE_WEIGHTS, CONTENT_WEIGHTS,LAMBDA1, LAMBDA2, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network')

            train(style_weight, content_weight, lambda1, lambda2, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)

        for style_weight, content_weight, lambda1, lambda2, model_save_path in zip(STYLE_WEIGHTS, CONTENT_WEIGHTS,LAMBDA1, LAMBDA2, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images')

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight) + '-' + str(content_weight) + '-' + str(lambda1) + '-' + str(lambda2))

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()