import tensorflow as tf
import numpy as np

def decoder(inputs, ifeatures, ofeatures, training, train_conv=True):
    with tf.name_scope('decoder'):
        x_c = tf.layers.conv2d(inputs=inputs, filters=ifeatures//4, kernel_size=1, strides=1, padding='same', trainable=train_conv)
        x_bn = tf.contrib.layers.batch_norm(inputs=x_c, scale=True, is_training=training)
        x_r = tf.nn.relu(x_bn)
        x_t = tf.layers.conv2d_transpose(inputs=x_r, filters=ifeatures//4, kernel_size=3, strides=2, padding='same', trainable=train_conv)
        x_bn2 = tf.contrib.layers.batch_norm(inputs=x_t, scale=True, is_training=training)
        x_r2 = tf.nn.relu(x_bn2)
        x_c2 = tf.layers.conv2d(inputs=x_r2, filters=ofeatures, kernel_size=1, strides=1, padding='same', trainable=train_conv)
        x_bn3 = tf.contrib.layers.batch_norm(inputs=x_c2, scale=True, is_training=training)
        x_r3 = tf.nn.relu(x_bn3)

    return x_r3

def max_pool_2x2(inputs, k_size, stride, name = None):
    return tf.layers.max_pooling2d(inputs, pool_size=k_size, strides=stride, padding='same', name='max_pool_' + name)

def conv_layer_with_ReLU(inputs, f_size, k_size, stride, name = None):
    x_c = tf.layers.conv2d(inputs=inputs, filters=f_size, kernel_size=k_size, strides=stride, padding='same', name="conv_" + name)
    x_bn = tf.contrib.layers.batch_norm(inputs=x_c, scale=True)
    x_r = tf.nn.relu(x_bn, name="relu_" + name)

    return x_r

def conv_layer_without_ReLU(inputs, f_size, k_size, stride, name = None):
    x_c = tf.layers.conv2d(inputs=inputs, filters=f_size, kernel_size=k_size, strides=stride, padding='same', name="conv_" + name)
    x_bn = tf.contrib.layers.batch_norm(inputs=x_c, scale=True)

    return x_bn

def conv_layer_with_ReLU_transpose(inputs, f_size, k_size, stride, name = None):
    x_c = tf.layers.conv2d_transpose(inputs=inputs, filters=f_size, kernel_size=k_size, strides=stride, padding='same', name="conv_" + name)
    x_bn = tf.contrib.layers.batch_norm(inputs=x_c, scale=True)
    x_r = tf.nn.relu(x_bn, name="relu_" + name)

    return x_r

def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv_layer_transpose_strided(x, W, b, output_shape=None, stride = 2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    
    return tf.nn.bias_add(conv, b)