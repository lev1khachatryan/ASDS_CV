'''
Useful functions, specific for all module parts of code 
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

# def conv2d(x, kernel, bias, use_relu=True):
#     '''
#     Convolution operation with reflect padding and valid.
#     Parameters
#     ----------
#     x : array_like
#         Four dimension tensor (batch_size, height, width, channels)
#     kernel : ndarray
#         Kernel we want to apply
#     bias : ndarray
#         bias we want to add
#     use_relu : bool, optional
#         Choose whether use relu or not 
#     Returns:
#     ----------
#     out : array_like
#         convoluted x with kernel and added by bias
#     '''

#     # padding image with reflection mode
#     x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

#     # conv and add bias
#     out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
#     # out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='SAME')
#     out = tf.nn.bias_add(out, bias)

#     if use_relu:
#         out = tf.nn.relu(out)

#     return out


def upsample(x, scale=2):
    '''
    Upsample the input image.
    Parameters
    ----------
    x : array_like
        Four dimension tensor (batch_size, height, width, channels)
    scale : int, float
        The amount of upsample 
    Returns:
    ----------
    output : array_like
        upsampled image by scale factor
    '''

    height = tf.shape(x)[1] * scale
    width  = tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output

def pool2d(x):
    '''
    Max pooling.
    Parameters
    ----------
    x : array_like
        Four dimension tensor (batch_size, height, width, channels)
    Returns:
    ----------
    array_like
        2D max pooled input x
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv(x, num_filter, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    '''
    Convolution operation with reflect padding and valid.
    Parameters
    ----------
    x : array_like
        Four dimension tensor (batch_size, height, width, channels)
    num_filter : int
        Number of filters in convolutional layer
    kernel : ndarray
        Kernel we want to apply
    bias : ndarray
        bias we want to add
    use_bias : bool, optional
        Choose whether use bias term or not 
    Returns:
    ----------
    x : array_like
        convoluted x with kernel and added by bias
    '''

    # with tf.variable_scope(scope):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    x = tf.layers.conv2d(inputs=x, filters=num_filter,
                         kernel_size=kernel, kernel_initializer=weight_init,
                         kernel_regularizer=weight_regularizer,
                         strides=stride, use_bias=use_bias, padding='SAME')
    return x

def hw_flatten(x) :
    '''
    Parameters
    ----------
    x : array_like
        Four dimension tensor (batch_size, height, width, channels)
    Returns
    ----------
    array_like
        reshaped input
    '''

    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])