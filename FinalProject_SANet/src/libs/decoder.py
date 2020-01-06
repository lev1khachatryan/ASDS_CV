'''
Decoder mostly mirrors the encoder with all pooling layers replaced 
by nearest up-sampling to reduce checker-board effects.
'''

import numpy as np
import tensorflow as tf
# from utils import *
from functions import *


class Decoder:
    '''
    Symmetric to encoder. 
    Decoder follows the settings of AdaIN 
    (Arbitrary style transfer in real-time with adaptive instance normalization)
    '''

    def __init__(self):
        '''
        Construct an symmetric architecture.
        Parameters
        ----------
        Returns:
        ----------
        '''

        self.weight_vars = []

        with tf.variable_scope('decoder'):
            self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
            self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

            self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(128,  64, 3, scope='conv2_1'))

            self.weight_vars.append(self._create_variables( 64,  64, 3, scope='conv1_2'))
            self.weight_vars.append(self._create_variables( 64,   3, 3, scope='conv1_1'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        '''
        Create simple variable and return it.
        Parameters
        ----------
        input_filters : int, float
            Number of input filters
        output_filters : int, float
            Number of output filters
        kernel_size : int, float
            shape of the kernel
        scope : string
            Tensorflow variable scope
        Returns:
        ----------
        kernel, bias : tuple
            Created variable
        '''

        with tf.variable_scope(scope):
            shape  = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape, name='kernel')
            bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=[output_filters], name='bias')
            return (kernel, bias)

    def decode(self, image):
        '''
        Decode the input.
        Parameters
        ----------
        image : array_like
            Four dimension tensor (batch_size, height, width, channels)
        Returns:
        ----------
        out : array_like
            Decoded input
        '''

        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 6)
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in upsample_indices:
                out = upsample(out)

        return out

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