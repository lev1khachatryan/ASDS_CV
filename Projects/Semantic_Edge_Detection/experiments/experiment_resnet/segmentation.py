import tensorflow as tf
from torch.utils import data
import sys
import os
import cv2
from skimage.io import imsave

# On Linux
# sys.path.insert(0, '/'.join(__file__.split('/')[:-3]))

# On Windows
sys.path.insert(0, '\\'.join(os.path.realpath(__file__).split('\\')[:-3]))

from dataloaders.segmentation_loader import seg_data, seg_data_test

from utils_resnet import *
from base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt

class SegModel(BaseModel):
    def __init__(self, args, sess):
        super(SegModel, self).__init__(args, sess)

        self.placeholders_dict['keep_prob'] = tf.placeholder(tf.float32, name='keep_prob')
        self.keep = self.placeholders_dict['keep_prob']
        self.placeholders_dict['train_batch_norms'] = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                                                  shape=(),
                                                                                  name='train_batch_norms')
        self.train_batch_norms = self.placeholders_dict['train_batch_norms']

        self.placeholders_dict['images'] = tf.placeholder(tf.float32, name='images',
                                                          shape=[None, self.image_height, self.image_width, 3])

        self.images = self.placeholders_dict['images']

        self.placeholders_dict['contours'] = tf.placeholder(tf.float32, name='contours',
                                                          shape=[None, self.image_height, self.image_width, 1])
        self.labels = self.placeholders_dict['contours']

        self.images_in_summary = ['images', 'contours']

        self.hyper_init_dict_train['train_batch_norms'] = True
        self.hyper_init_dict_train['learning_rate'] = self.lr
        self.hyper_init_dict_train['keep_prob'] = 0.7

        self.hyper_init_dict_test_val['train_batch_norms'] = False
        self.hyper_init_dict_test_val['learning_rate'] = self.lr
        self.hyper_init_dict_test_val['keep_prob'] = 1.0

        self.updates = [] # for batch_norm updates

        if self.mode not in ['freeze', 'test']:
            self.dataloader_train = data.DataLoader(seg_data(root=os.path.join(self.dataset_path, 'train'),
                                                             image_height=self.image_height,
                                                             image_width=self.image_width),
                                                    batch_size=self.batch_size,
                                                    num_workers=4,
                                                    shuffle=False)

            self.dataloader_val = data.DataLoader(
                                                    seg_data(root=os.path.join(self.dataset_path, 'test'),
                                                                               image_height=self.image_height,
                                                                               image_width=self.image_width),
                                                    batch_size=self.batch_size,
                                                    num_workers=4,
                                                    shuffle=False)
            self.dataloader_test = data.DataLoader(
                                                    seg_data(root=os.path.join(self.dataset_path, 'test'),
                                                                               image_height=self.image_height,
                                                                               image_width=self.image_width,
                                                                               mode='test'),
                                                    batch_size=1,
                                                    num_workers=4,
                                                    shuffle=False)
        elif self.mode == 'test':
            self.dataloader_test_visual = data.DataLoader(seg_data_test(root=args.test_dir,
                                                                        image_height=self.image_height,
                                                                        image_width=self.image_width),
                                                          batch_size=1,
                                                          num_workers=4,
                                                          shuffle=False)

    def preprocess_inputs(self, inputs_to_preprocess):
        preprocessed_dict = dict()

        preprocessed_dict['images'] = self.mean_image_subtraction(inputs_to_preprocess['images'])

        preprocessed_dict['keep_prob'] = self.keep
        preprocessed_dict['train_batch_norms'] = self.train_batch_norms
        return preprocessed_dict

    def mean_image_subtraction(self, image):
        means = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
            channels[i] /= std[i]
        return tf.concat(axis=3, values=channels)

    def get_optimizer(self, learning_rate, var_list_to_update):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(self.totall_loss,
                                                                                var_list=var_list_to_update)
        return [train_op] + self.updates

    def get_losses(self, ground_truths, predictions):
        
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predictions, labels = ground_truths))
        # mse = tf.reduce_mean(tf.squared_difference(predictions, ground_truths))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.softmax, labels = self.placeholders_dict['contours']))
        mse = tf.reduce_mean(tf.squared_difference(self.placeholders_dict['contours'], self.softmax))

        lambda_1 = lambda_2 = 0.5

        totall_loss = lambda_1 * loss + lambda_2 * mse
        losses_dict = {'cross_entropy': loss, 'totall_loss': totall_loss, 'mse': mse}
        return totall_loss, losses_dict


    def build_graph(self, input_dict, mode, num_classes):
        outs_dict = self.network(input_dict, mode, num_classes)
        return outs_dict

    def network(self, input_dict, mode, num_classes, batch_size=None, use_l2_regularizer=True):
    	
        input_shape = (224, 224, 3)
        img_input = layers.Input(shape=input_shape, batch_size=batch_size)

        x = img_input
        bn_axis = 3

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
        x = layers.Conv2D(
          64, (7, 7),
          strides=(2, 2),
          padding='valid',
          use_bias=False,
          kernel_initializer='he_normal',
          kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
          name='conv1')(
              x)
        x = layers.BatchNormalization(
          axis=bn_axis,
          momentum=BATCH_NORM_DECAY,
          epsilon=BATCH_NORM_EPSILON,
          name='bn_conv1')(
              x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = conv_block(
          x,
          3, [64, 64, 256],
          stage=2,
          block='a',
          strides=(1, 1),
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [64, 64, 256],
          stage=2,
          block='b',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [64, 64, 256],
          stage=2,
          block='c',
          use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
          x,
          3, [128, 128, 512],
          stage=3,
          block='a',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [128, 128, 512],
          stage=3,
          block='b',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [128, 128, 512],
          stage=3,
          block='c',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [128, 128, 512],
          stage=3,
          block='d',
          use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='a',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='b',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='c',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='d',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='e',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [256, 256, 1024],
          stage=4,
          block='f',
          use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
          x,
          3, [512, 512, 2048],
          stage=5,
          block='a',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [512, 512, 2048],
          stage=5,
          block='b',
          use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
          x,
          3, [512, 512, 2048],
          stage=5,
          block='c',
          use_l2_regularizer=use_l2_regularizer)



        #-----------------------------Build fully connvolutional layers-----------------------------
        with tf.name_scope('fully_connected'):
            self.conv6 = conv_layer_with_ReLU(inputs=x, f_size=4096, k_size=9, stride=1, name="6")
            # self.dropout6 = tf.nn.dropout(self.conv6, keep_prob=keep_prob)

            self.conv7 = conv_layer_with_ReLU(inputs=self.conv6, f_size=4096, k_size=1, stride=1, name="7")
            # self.dropout7 = tf.nn.dropout(self.conv7, keep_prob=keep_prob)

            self.conv8 = conv_layer_with_ReLU(inputs=self.conv7, f_size=num_classes, k_size=1, stride=1, name="8")


        #-----------------------------Build network decoder-----------------------------
        with tf.name_scope('decoder'):
            # deconv_shape1 = self.pool4.get_shape()
            # self.conv_t1 = conv_layer_with_ReLU_transpose(inputs=self.conv8, f_size=num_classes, k_size=4, stride=1, name="t1")

            # self.conv9 = conv_layer_without_ReLU(inputs=self.conv_t1, f_size=num_classes, k_size=1, stride=1, name="9")

            deconv_shape1 = self.pool4.get_shape()
            W_t1 = weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
            b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
            self.conv_t1 = conv_layer_transpose_strided(self.conv8, W_t1, b_t1, output_shape=tf.shape(self.pool4))
            self.fuse_1 = tf.add(self.conv_t1, self.pool4, name="fuse_1")

            deconv_shape2 = self.pool3.get_shape()
            W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
            self.conv_t2 = conv_layer_transpose_strided(self.fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
            self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")

            shape = tf.shape(input_dict['images'])
            W_t3 = weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
            b_t3 = bias_variable([num_classes], name="b_t3")

            out = conv_layer_transpose_strided(self.fuse_2, W_t3, b_t3, output_shape=[shape[0], shape[1], shape[2], num_classes], stride=8)

        pred = tf.argmax(out, axis=-1)
        # pred = tf.argmax(out, dimension=3, name="pred")
        
        softmax = tf.nn.softmax(out)

        self.logits = out
        self.softmax = tf.identity(softmax, name='softmax')
        self.images_in_summary += ['softmax']

        return {'predictions':pred, 'softmax':softmax, 'logits':out}

    def compute_IOU(self, preds_dict, gt_dict):
        preds = preds_dict['predictions']
        labels = gt_dict['contours']

        labels[labels > 0.5] = 1
        labels[labels < 0.5] = 0

        ious = []
        for smpl_id in range(len(preds)):
            pred = preds[smpl_id]
            label = labels[smpl_id, :, :, 0]
            # TODO implement the metric Intersection Over Union
            iou = 0
            ious.append(iou)
        return {'mean_IOU': np.mean(ious)}

    def blending(self, preds_dict, gt_dict, name=None, path_to_save=None):
        color_path = '/'.join(os.path.dirname(__file__).split('/')[:-2])+'/bg_green.jpg'
        if not os.path.exists(color_path):
            color = np.stack([np.zeros((1080, 1920)), np.ones((1080, 1920)), np.zeros((1080, 1920))], axis = -1)
            color = np.array(color*255, dtype=np.uint8)
            imsave(color_path, color)
            
        img = gt_dict['images'][0, :, :, :]
        contour = preds_dict['softmax'][0, :, :, 1]

        bg = cv2.imread('bg_green.jpg')[:, :, ::-1] / 255.
        height = img.shape[0]
        width = img.shape[1]

        img = cv2.resize(img, (width, height))
        contour = cv2.resize(contour, (width, height), interpolation=cv2.INTER_NEAREST)

        bg_crop = cv2.resize(bg, (width, height))

        img[:, :, 0] = img[:, :, 0] * contour + bg_crop[:, :, 0] * (1 - contour)
        img[:, :, 1] = img[:, :, 1] * contour + bg_crop[:, :, 1] * (1 - contour)
        img[:, :, 2] = img[:, :, 2] * contour + bg_crop[:, :, 2] * (1 - contour)

        imsave(os.path.join(path_to_save, name + '.png'), np.clip(img, 0, 1))
        return img

    def get_prediction_results(self, preds_dict, gt_dict, name=None, path_to_save=None):
        contour = preds_dict['predictions'][0, :, :]
        imsave(os.path.join(path_to_save, name + '.png'), contour)
        return contour