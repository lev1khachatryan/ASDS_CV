import numpy as np
import tensorflow as tf
import scipy.misc
from torch.utils import data
import os
from tqdm import tqdm
import time
import pprint
import sys

try:
    import matplotlib.pyplot as plt
except:
    print('Can not import matplotlib')

from tensorflow.python import graph_util
from skimage.io import imsave


def torch_to_numpy(torch_tensor):
    if len(torch_tensor[0].shape) == 4:
        return np.transpose(torch_tensor[0].numpy(), [0, 2, 3, 1])
    else:
        return torch_tensor[0].numpy()


class BaseModel(object):
    def __init__(self, args, sess):
        self.dataset_path = args.dataset_path
        self.checkpoint_dir = args.checkpoint_dir
        self.summary_dir = args.summaries_dir

        self.batch_size = args.batch_size
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.num_classes = args.num_classes
        self.epoch = args.epoch
        self.lr = args.lr
        self.mode = args.mode

        self.sess = sess

        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.placeholders_dict = {'learning_rate': self.learning_rate}
        # dicts for hyperparametric placeholders such as keep_prob, train_batch_norm, ...
        self.hyper_init_dict_train = dict()
        self.hyper_init_dict_test_val = dict()
        self.images_in_summary = []
        self.scopes = []  # List of variables which will be trainable in spite of being loaded from checkpoint

        self.summary_write_freq = args.summary_write_freq

        self.dataloader_train = None

        self.dataloader_val = None

        # For measuring metrics
        self.dataloader_test = None

        self.dataloader_test_visual = None

    def build_model(self):
        ins_dict = self.preprocess_inputs(self.placeholders_dict)

        self.outs_dict = self.build_graph(ins_dict, self.mode, self.num_classes)

        # self.totall_loss, self.losses_dict = self.get_losses(self.placeholders_dict, self.outs_dict)
        self.totall_loss, self.losses_dict = self.get_losses()

        self.summary_op, self.summary_writer_train, self.summary_writer_val = self.get_summaries(self.placeholders_dict,
                                                                                                 self.outs_dict,
                                                                                                 self.losses_dict,
                                                                                                 self.images_in_summary)

        self.saver = tf.train.Saver(max_to_keep=10000)

        _, var_list = self.get_uninitialized_variables()

        for scope in self.scopes:
            var_list = var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        var_list = list(set(var_list))  # otherwise in optimizer some variables may be updated more than one time

        self.train_op = self.get_optimizer(self.learning_rate, var_list)

        # Initialize uninitialized variables and reading from checkpoint if checkpoint exists.
        self.initialize_network()

    def train(self):
        for epoch in tqdm(range(self.epoch)):
            np.random.uniform(0., 1.)
            # iterator for validation
            self.dataloader_val_iter = iter(self.dataloader_val)

            # training process
            for idx, f_dict_train in tqdm(enumerate(self.dataloader_train)):
                assert isinstance(f_dict_train, dict)
                start_time = time.time()

                feed_dict_train = dict()
                assert set(f_dict_train.keys()).union(set(self.hyper_init_dict_train.keys())) == \
                       set(self.placeholders_dict.keys())

                for key in self.placeholders_dict.keys():
                    if key in f_dict_train.keys():
                        feed_dict_train[self.placeholders_dict[key]] = torch_to_numpy(f_dict_train[key])
                    else:
                        feed_dict_train[self.placeholders_dict[key]] = self.hyper_init_dict_train[key]

                _, summary, train_loss = self.sess.run([self.train_op, self.summary_op, self.totall_loss],
                                                       feed_dict=feed_dict_train)

                if idx % self.summary_write_freq == 0:
                    self.summary_writer_train.add_summary(summary, idx)
                    print("Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f" %
                          (epoch, idx, time.time() - start_time, train_loss))

                if idx % (2 * self.summary_write_freq) == 0:
                    try:
                        f_dict_val = next(self.dataloader_val_iter)
                        assert isinstance(f_dict_val, dict)

                        feed_dict_val = dict()
                        for key in self.placeholders_dict.keys():
                            if key in f_dict_val.keys():
                                feed_dict_val[self.placeholders_dict[key]] = torch_to_numpy(f_dict_val[key])
                            else:
                                feed_dict_val[self.placeholders_dict[key]] = self.hyper_init_dict_test_val[key]

                        summary_val, val_loss = self.sess.run([self.summary_op, self.totall_loss],
                                                              feed_dict=feed_dict_val)

                        self.summary_writer_val.add_summary(summary_val, idx)
                        print("VALIDATION Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f"
                              % (epoch, idx, time.time() - start_time, val_loss))

                    except StopIteration:
                        self.dataloader_val_iter = iter(self.dataloader_val)

            # if epoch % 100 == 0:
            self.save(self.checkpoint_dir, epoch)

    def measure_metrics(self, args, metric):
        metric_epoch_dict = dict()
        for epoch in range(args.test_start, args.test_end):
            if not self.load_custom_checkpoint(self.checkpoint_dir, epoch):
                print(" [!] Load for measuring metrics failed...")

            metric_list = []
            for idx, f_dict_test in enumerate(self.dataloader_test):
                feed_dict_test = dict()
                gt_dict_test = {key:torch_to_numpy(f_dict_test[key]) for key in f_dict_test.keys()}

                assert set(f_dict_test.keys()).union(set(self.hyper_init_dict_test_val.keys())) == \
                       set(self.placeholders_dict.keys())

                for key in self.placeholders_dict.keys():
                    if key in f_dict_test.keys():
                        feed_dict_test[self.placeholders_dict[key]] = torch_to_numpy(f_dict_test[key])
                    else:
                        feed_dict_test[self.placeholders_dict[key]] = self.hyper_init_dict_test_val[key]


                preds_dict = dict()
                out_keys = list(self.outs_dict.keys())

                pred_outs = self.sess.run([self.outs_dict[out_keys[i]] for i in range(len(out_keys))],
                                           feed_dict = feed_dict_test)

                for i in range(len(out_keys)):
                    preds_dict[out_keys[i]] = pred_outs[i]

                metric_list.append(metric(preds_dict, gt_dict_test))

            keys = metric_list[0].keys()
            metric_dict = {key:np.mean([x[key] for x in metric_list]) for key in keys}
            metric_epoch_dict['epoch '+ str(epoch)] = metric_dict
            print('metrics for epoch -'+str(epoch))
            pprint.pprint(metric_dict, width = 1)
        return metric_epoch_dict

    def test(self, path_to_save, epoch, out_keys_to_save, function):
        if not self.load_custom_checkpoint(self.checkpoint_dir, epoch):
            print(" [!] Load for visual testing failed...")
            exit()

        for idx, f_dict_test in tqdm(enumerate(self.dataloader_test_visual)):
            assert 'name' in f_dict_test.keys()
            feed_dict_test = dict()

            gt_dict_vis = {key: torch_to_numpy(f_dict_test[key]) for key in f_dict_test.keys()
                           if key != 'name'}

            for key in self.placeholders_dict.keys():
                if key in f_dict_test.keys():
                    feed_dict_test[self.placeholders_dict[key]] = torch_to_numpy(f_dict_test[key])
                else:
                    feed_dict_test[self.placeholders_dict[key]] = self.hyper_init_dict_test_val[key]

            preds_dict = dict()

            pred_outs = self.sess.run([self.outs_dict[out_keys_to_save[i]] for i in range(len(out_keys_to_save))],
                                      feed_dict=feed_dict_test)

            for i in range(len(out_keys_to_save)):
                preds_dict[out_keys_to_save[i]] = pred_outs[i]

            function(preds_dict, gt_dict_vis, name=f_dict_test['name'][0], path_to_save=path_to_save)

    def preprocess_inputs(self, inputs_to_preprocess):
        raise NotImplementedError

    def build_graph(self, input_dict, mode, num_classes):
        return self.network(input_dict, mode, num_classes)

    def network(self, input_dict, mode, num_classes):
        raise NotImplementedError

    def get_losses(self, ground_truths, predictions):
        raise NotImplementedError

    def get_summaries(self, ins_dict, outs_dict, losses_dict, images_in_summary):
        for key in ins_dict.keys():
            if key in images_in_summary:
                if len(ins_dict[key].shape) == 4:
                    if ins_dict[key].dtype == 'int32':
                        summary_show = tf.cast(ins_dict[key] * int(255. / (self.num_classes - 1)), dtype=tf.uint8)
                    else:
                        summary_show = ins_dict[key]
                    tf.summary.image(key, summary_show, max_outputs=2)

        for key in outs_dict.keys():
            if key in images_in_summary:
                if len(outs_dict[key].shape) == 4:
                    if outs_dict[key].shape[-1] != 3:
                        summary_show = tf.expand_dims(outs_dict[key][:, :, :, 0], axis=-1)
                    else:
                        summary_show = outs_dict[key]
                    if summary_show.dtype == 'int32':
                        summary_show = tf.cast(summary_show * int(255. / (self.num_classes - 1)), dtype=tf.uint8)
                    tf.summary.image(key, summary_show, max_outputs=2)

        for key in losses_dict.keys():
            tf.summary.scalar(key, losses_dict[key])

        summary_op = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter(self.summary_dir + '/train', self.sess.graph)

        summary_writer_val = tf.summary.FileWriter(self.summary_dir + '/validation', self.sess.graph)

        return summary_op, summary_writer_train, summary_writer_val

    def get_optimizer(self, learning_rate, var_list_to_update):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.totall_loss,
                                                                         var_list=var_list_to_update)
        return train_op

    def initialize_network(self):
        vars_initializer, _ = self.get_uninitialized_variables()
        vars_initializer.run()

        if self.load(self.checkpoint_dir, self.saver):
            print(" [*] Loading from checkpoint SUCCEEDED")
        else:
            print(" [!] Loading from checkpoint failed...")

    def get_uninitialized_variables(self):
        glb = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if sys.version_info.major == 2:
            uninitialized_variables = set(self.sess.run(tf.report_uninitialized_variables()))
        elif sys.version_info.major == 3:
            uninitialized_variables = set(
                [var.decode('utf-8') for var in self.sess.run(tf.report_uninitialized_variables())])

        var_list = []
        for v in glb:
            if v.name.split(':')[0] in uninitialized_variables:
                var_list.append(v)
        return tf.variables_initializer(var_list), var_list

    def load(self, checkpoint_dir, saver):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(ckpt_name)
            return True
        else:
            return False

    def load_custom_checkpoint(self, checkpoint_dir, index):
        print('loading from checkpoint ' + str(index) + ' ...')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if index == -1:
            index = int(ckpt.model_checkpoint_path.split('-')[-1])
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_name = ckpt_name.split('-')[0] + '-' + str(index)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print(ckpt_name)
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        model_name = "matting.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def freeze_save_graph(self, sess, output_nodes, path, name):
        for node in sess.graph.as_graph_def().node:
            node.device = ""
        variable_graph_def = sess.graph.as_graph_def()
        optimized_net = graph_util.convert_variables_to_constants(sess, variable_graph_def, output_nodes)
        tf.train.write_graph(optimized_net, path, name, False)