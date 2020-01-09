'''
Style-Attentional Network learns the mapping
between the content features and the style features
'''

import numpy as np
import tensorflow as tf
# import tensorflow.contrib as tf_contrib
# import sys
# sys.path.append('../../libs')
# sys.path.insert(1, '../')
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
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            f = conv(content, self.num_filter, kernel=1, stride=1, scope='f_conv') # [bs, h, w, c']
            g = conv(style,   self.num_filter, kernel=1, stride=1, scope='g_conv') # [bs, h, w, c']
            h = conv(style,   self.num_filter, kernel=1, stride=1, scope='h_conv') # [bs, h, w, c]

            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True, name='mm1') # [bs, N, N]  N = h * w

            attention = tf.nn.softmax(s)  # attention map

            o = tf.matmul(hw_flatten(h), attention, transpose_a=True, name='mm2') # [bs, N, C]
            o = tf.reshape(o, shape=tf.shape(content)) # [bs, h, w, C]
            o = conv(o, self.num_filter, kernel=1, stride=1, scope='attn_conv')

            o = o + content

            return o