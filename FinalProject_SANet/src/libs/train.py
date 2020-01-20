'''
Train the Style Transfer Net
'''

from __future__ import print_function

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf

import numpy as np
import tensorflow as tf

from stnet import STNet
from utils import get_train_images


STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

TRAINING_IMAGE_SHAPE = (512, 512, 3) # (height, width, color_channels)

EPOCHS = 1000
EPSILON = 1e-5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0


def train(style_weight, content_weight, lambda1, lambda2, content_imgs_path, style_imgs_path, encoder_path, 
          model_save_path, debug=True, logging_period=10):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # guarantee the count of content and style images to be a multiple of BATCH_SIZE
    num_imgs = min(len(content_imgs_path), len(style_imgs_path))
    content_imgs_path = content_imgs_path[:num_imgs]
    style_imgs_path   = style_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        content_imgs_path = content_imgs_path[:-mod]
        style_imgs_path   = style_imgs_path[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

        # create the style transfer net
        stn = STNet(encoder_path)

        # get the target feature maps which is the output of SAModule
        # Fcsc_m = stn.Fcsc_m

        # pass content and style to the stn, getting the Ics (generated image)
        Ics = stn.transform(content, style)

        # pass the Ics to the encoder, and use the output compute loss
        # Ics = tf.reverse(Ics, axis=[-1])  # switch RGB to BGR
        # Ics = stn.encoder.preprocess(Ics) # preprocess image
        Ics_enc = stn.encoder.encode(Ics)

        # compute the content loss
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(Ics_enc['relu4_1'] - stn.encoded_content_layers['relu4_1']), axis=[1, 2]) + \
        							 tf.reduce_mean(tf.square(Ics_enc['relu5_1'] - stn.encoded_content_layers['relu5_1']), axis=[1, 2]))

        # compute the style loss
        style_layer_loss = []
        for layer in STYLE_LAYERS:
            enc_style_feat = stn.encoded_style_layers[layer]
            enc_gen_feat   = Ics_enc[layer]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + EPSILON)
            sigmaG = tf.sqrt(varG + EPSILON)

            l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            # l2_mean  = tf.reduce_mean(tf.square(meanG - meanS))
            # l2_sigma = tf.reduce_mean(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # compute the Identity loss lambda 1
        Icc = stn.transform(content, content)
        # Icc = tf.reverse(Icc, axis=[-1])
        # Icc = stn.encoder.preprocess(Icc)
        Iss = stn.transform(style, style)
        # Iss = tf.reverse(Iss, axis=[-1])
        # Iss = stn.encoder.preprocess(Iss)
        loss_lambda1 = tf.reduce_sum(tf.reduce_mean(tf.square(Icc - content), axis=[1, 2]) + \
                                     tf.reduce_mean(tf.square(Iss - style), axis=[1, 2]))

        # compute the Identity loss lambda 2
        Icc_enc = stn.encoder.encode(Icc)
        Iss_enc = stn.encoder.encode(Iss)
        # loss_lambda2 = tf.reduce_sum(tf.reduce_mean(tf.square(Icc_enc['relu1_1'] - stn.encoded_content_layers['relu1_1']), axis=[1, 2]) +
        #                              tf.reduce_mean(tf.square(Iss_enc['relu1_1'] - stn.encoded_style_layers['relu1_1']), axis=[1, 2]))
        loss_lambda2 = []
        for layer in STYLE_LAYERS:
            loss_lambda2_ = tf.reduce_mean(tf.square(Icc_enc[layer] - stn.encoded_content_layers[layer])) + \
                            tf.reduce_mean(tf.square(Iss_enc[layer] - stn.encoded_style_layers[layer]))

            loss_lambda2.append(loss_lambda2_)

        loss_lambda2 = tf.reduce_sum(loss_lambda2)



        # compute the total loss
        loss =  content_weight*content_loss + style_weight*style_loss + lambda1*loss_lambda1 + lambda2 * loss_lambda2

        # Training step
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        # saver
        saver = tf.train.Saver(max_to_keep=10)

        ###### Start Training ######
        step = 0
        n_batches = int(len(content_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            start_time = datetime.now()
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')

        try:
            for epoch in range(EPOCHS):
                np.random.shuffle(content_imgs_path)
                np.random.shuffle(style_imgs_path)

                for batch in range(n_batches):
                    # retrive a batch of content and style images
                    content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                    style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                    content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                    style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

                    # run the training step
                    sess.run(train_op, feed_dict={content: content_batch, style: style_batch})

                    step += 1

                    if step % 50 == 0:
                        saver.save(sess, model_save_path, global_step=step, write_meta_graph=False)

                    if debug:
                        is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                        if is_last_step or step == 1 or step % logging_period == 0:
                            elapsed_time = datetime.now() - start_time
                            _content_loss, _style_loss, _loss_lambda1, _loss_lambda2, _loss = sess.run([content_loss, style_loss, loss_lambda1, loss_lambda2, loss], 
                                feed_dict={content: content_batch, style: style_batch})

                            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                            print('content loss: %.3f,  weighted content loss: %.3f' % (_content_loss, content_weight * _content_loss))
                            print('style loss  : %.3f,  weighted style loss: %.3f' % (_style_loss, style_weight * _style_loss))
                            print('lambda1 loss  : %.3f,  weighted lambda1 loss: %.3f' % (_loss_lambda1, lambda1 * _loss_lambda1))
                            print('lambda2 loss  : %.3f,  weighted lambda2 loss: %.3f\n' % (_loss_lambda2, lambda2 * _loss_lambda2))
        except Exception as ex:
            saver.save(sess, model_save_path, global_step=step)
            print('\nSomething wrong happens! Current model is saved to <%s>' % tmp_save_path)
            print('Error message: %s' % str(ex))

        ###### Done Training & Save the model ######
        saver.save(sess, model_save_path)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % model_save_path)