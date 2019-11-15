import tensorflow as tf

def decoder(inputs, ifeatures, ofeatures, training, train_conv=True):
    with tf.name_scope('decoder'):
        x_c = tf.layers.conv2d(inputs=inputs, filters=ifeatures//4, kernel_size=1, strides=1, padding='same', trainable=train_conv)
        x_bn = tf.contrib.layers.batch_norm(inputs=x_c, scale=True, is_training=training)
        x_r = tf.nn.relu(x_bn)
        x_t = tf.layers.conv2d_transpose(inputs=x_r, filters=ifeatures//4, kernel_size=3, strides=2, padding='same', trainable=train_conv)
        x_bn2 = tf.contrib.layers.batch_norm(inputs=x_t, scale=True, is_training=training)
        x_r2 = tf.nn.relu(x_bn2)
        x_c2 = tf.layers.conv2d(inputs=x_r2, filters=ofeatures, kernel_size=1, strides=1,
                                padding='same', trainable=train_conv)
        x_bn3 = tf.contrib.layers.batch_norm(inputs=x_c2, scale=True, is_training=training)
        x_r3 = tf.nn.relu(x_bn3)

    return x_r3
