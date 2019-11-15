import tensorflow as tf
from torch.utils import data
import sys
import os
import cv2
from skimage.io import imsave

sys.path.insert(0, '/'.join(__file__.split('/')[:-3]))
from dataloaders.segmentation_loader import seg_data, seg_data_test
from utils import decoder
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

        self.placeholders_dict['masks'] = tf.placeholder(tf.float32, name='masks',
                                                          shape=[None, self.image_height, self.image_width, 1])
        self.labels = self.placeholders_dict['masks']

        self.images_in_summary = ['images', 'masks']

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
                                                    shuffle=True)

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
        # TODO implement the totall loss as the weighted sum of two losses, namely cross entropy loss and MSE
        loss = 0
        mse = 0

        # TODO tune the hyperparameters lambda_1 and lambda_2
        lambda_1 = lambda_2 = 1

        totall_loss = lambda_1 * loss + lambda_2 * mse
        losses_dict = {'cross_entropy': loss, 'totall_loss': totall_loss, 'mse': mse}
        return totall_loss, losses_dict


    def build_graph(self, input_dict, mode, num_classes):
        outs_dict = self.network(input_dict, mode, num_classes)
        return outs_dict

    def network(self, input_dict, mode, num_classes):
        # TODO build an architecture

        out = None # TODO the last layer with 'num_classes' channels
        pred = tf.argmax(out, axis=-1)
        softmax = None # TODO apply softmax to get the distributions for each pixel

        self.logits = out
        self.softmax = tf.identity(softmax, name='softmax')
        self.images_in_summary += ['softmax']

        return {'predictions':pred, 'softmax':softmax, 'logits':out}

    def compute_IOU(self, preds_dict, gt_dict):
        preds = preds_dict['predictions']
        labels = gt_dict['masks']

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
        mask = preds_dict['softmax'][0, :, :, 1]

        bg = cv2.imread('bg_green.jpg')[:, :, ::-1] / 255.
        height = img.shape[0]
        width = img.shape[1]

        img = cv2.resize(img, (width, height))
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        bg_crop = cv2.resize(bg, (width, height))

        img[:, :, 0] = img[:, :, 0] * mask + bg_crop[:, :, 0] * (1 - mask)
        img[:, :, 1] = img[:, :, 1] * mask + bg_crop[:, :, 1] * (1 - mask)
        img[:, :, 2] = img[:, :, 2] * mask + bg_crop[:, :, 2] * (1 - mask)

        imsave(os.path.join(path_to_save, name + '.png'), np.clip(img, 0, 1))
        return img

    def get_prediction_results(self, preds_dict, gt_dict, name=None, path_to_save=None):
        mask = preds_dict['predictions'][0, :, :]
        imsave(os.path.join(path_to_save, name + '.png'), mask)
        return mask