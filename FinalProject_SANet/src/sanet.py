import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import *


class SANet:
    '''
    Style-Attentional Network learns the mapping 
    between the content features and the style features 
    by slightly modifying the self-attention mechanism
    '''

    def __init__(self, num_filter):
        self.num_filter = num_filter

    def map(self, content, style, scope='attention'):
        with tf.variable_scope(scope):
            f = conv(content, self.num_filter // 8, kernel=1, stride=1, scope='f_conv') # [bs, h, w, c']
            g = conv(style,   self.num_filter // 8, kernel=1, stride=1, scope='g_conv') # [bs, h, w, c']
            h = conv(style,   self.num_filter     , kernel=1, stride=1, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            attention = tf.nn.softmax(s)  # attention map

            o = tf.matmul(attention, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=content.shape) # [bs, h, w, C]
            o = conv(o, self.num_filter, kernel=1, stride=1, scope='attn_conv')

            o = gamma * o + content

            return o

def conv(x, num_filter, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
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
                             strides=stride, use_bias=use_bias)
        return x

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])