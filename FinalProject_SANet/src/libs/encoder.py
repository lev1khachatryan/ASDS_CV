'''
Encoder is fixed to the first few layers (up to relu5_1) 
of VGG-19 (pre-trained on ImageNet)
'''

import numpy as np
import tensorflow as tf
# from utils import *
from functions import *


ENCODER_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 
    'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
)

class Encoder:
    '''
    In this work, the pre-trained VGG-19 network is employed as encoder.
    '''

    def __init__(self, weights_path):
        '''
        Load vgg19 weights from npz file.
        Parameters
        ----------
        weights_path : string
            Relatrive path to pretrained weights
        Returns:
        ----------
        None
            The aim is to create object with appropriate fields
        '''

        # load weights from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []

        # create the TensorFlow variables
        with tf.variable_scope('encoder'):
            for layer in ENCODER_LAYERS:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx += 2

                    with tf.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):
        '''
        Create the computational graph and return weights values.
        Parameters
        ----------
        image : array_like
            Four dimension tensor (batch_size, height, width, channels)
        Returns:
        ----------
        layers : dictionary
            Keys are the name of layers and values are the pretrained weights.
        '''

        # create the computational graph
        idx = 0
        layers = {}
        current = image

        for layer in ENCODER_LAYERS:
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel, bias, use_relu=False)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current

        assert(len(layers) == len(ENCODER_LAYERS))

        return layers

    # def preprocess(self, image, mode='BGR'):
    #     '''
    #     Normalize the image.
    #     Parameters
    #     ----------
    #     image : array_like
    #         Four dimension tensor (batch_size, height, width, channels)
    #     mode : string, optional
    #         By default, assumed that images are reversed from RGB to BGR. 
    #     Returns:
    #     ----------
    #     image : array_like
    #         Normalized array
    #     '''

    #     if mode == 'BGR':
    #         return image - np.array([103.939, 116.779, 123.68])
    #     else:
    #         return image - np.array([123.68, 116.779, 103.939])

    # def deprocess(self, image, mode='BGR'):
    #     '''
    #     Denormalize the image.
    #     Parameters
    #     ----------
    #     image : array_like
    #         Four dimension tensor (batch_size, height, width, channels)
    #     mode : string, optional
    #         By default, assumed that images are reversed from RGB to BGR. 
    #     Returns:
    #     ----------
    #     image : array_like
    #         Denormalized array
    #     '''
    #     if mode == 'BGR':
    #         return image + np.array([103.939, 116.779, 123.68])
    #     else:
    #         return image + np.array([123.68, 116.779, 103.939])

def conv2d(x, kernel, bias, use_relu=True):
    '''
    Convolution operation with reflect padding and valid.
    Parameters
    ----------
    x : array_like
        Four dimension tensor (batch_size, height, width, channels)
    kernel : ndarray
        Kernel we want to apply
    bias : ndarray
        bias we want to add
    use_relu : bool, optional
        Choose whether use relu or not 
    Returns:
    ----------
    out : array_like
        convoluted x with kernel and added by bias
    '''

    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    # out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out