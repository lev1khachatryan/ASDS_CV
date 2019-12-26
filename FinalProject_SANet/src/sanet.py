import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import *
from functions import *


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