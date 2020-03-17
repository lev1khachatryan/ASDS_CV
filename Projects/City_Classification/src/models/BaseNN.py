import tensorflow as tf
from data_loader import *
from utils import *
from abc import abstractmethod
import numpy as np
import pandas as pd
import os;
import datetime  
import cv2
import os

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, inference_dir, num_epochs, train_batch_size,
                 val_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, height_of_image, width_of_image, num_channels, num_classes)

        self.inference_dir = inference_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.base_dir = base_dir
        self.max_to_keep = max_to_keep
        self.model_name = model_name
        self.keep_prob = keep_prob

        self.n_log_step = 0 # counting current number of mini batches trained on
    
    def create_network(self):
        """
        Create base components of the network.
        Main structure of network will be described in network function.
        -----------------
        Parameters:
            None
        Returns:
            None
        -----------------
        """
        tf.reset_default_graph()

        # variables for input and output 
        self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.height_of_image, self.data_loader.width_of_image, self.data_loader.num_channels], name='x_data_tf')
        self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.num_classes], name='y_data_tf')

        self.z_pred_tf = self.network(self.x_data_tf)

        # cost function
        self.cross_entropy_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_data_tf, logits=self.z_pred_tf), name = 'cross_entropy_tf')

        # optimisation function
        self.learn_rate_tf = tf.placeholder(dtype=tf.float32, name="learn_rate_tf")
        self.train_step_tf = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
            self.cross_entropy_tf, name = 'train_step_tf')

        # predicted probabilities in one-hot encoding
        self.y_pred_proba_tf = tf.nn.softmax(self.z_pred_tf, name='y_pred_proba_tf')
        
        # tensor of correct predictions
        self.y_pred_correct_tf = tf.equal(tf.argmax(self.y_pred_proba_tf, 1),
                                          tf.argmax(self.y_data_tf, 1),
                                          name = 'y_pred_correct_tf')  
        
        # accuracy 
        self.accuracy_tf = tf.reduce_mean(tf.cast(self.y_pred_correct_tf, dtype=tf.float32),
                                         name = 'accuracy_tf')

        # tensors to save intermediate accuracies and losses during training
        self.train_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                         name='train_loss_tf', validate_shape = False)
        self.valid_loss_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                         name='valid_loss_tf', validate_shape = False)
        self.train_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                        name='train_acc_tf', validate_shape = False)
        self.valid_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                        name='valid_acc_tf', validate_shape = False)
        return None

    def close_writers(self):
        """
        Close train and validation summary writer.
        -----------------
        Parameters:
            sess - the session we want to save
        Returns:
            None
        -----------------
        """
        self.train_writer.close()
        self.valid_writer.close()

        return None

    def save_model(self, sess):
        """
        Save tensors/summaries
        -----------------
        Parameters:
            sess - the session we want to save
        Returns:
            None
        -----------------
        """
        filepath = os.path.join(self.base_dir, self.model_name, 'checkpoints', self.model_name)
        self.saver_tf.save(sess, filepath)
        
        return None

    def train_model_helper(self, sess, n_epoch = 1):
        """
        Helper function to train the model.
        -----------------
        Parameters:
            sess - current session
            n_epoch (int)         - number of epochs
        Returns:
            None
        -----------------
        """
        
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        # start timer
        start = datetime.datetime.now()
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ', self.learning_rate,', n_epoch = ', n_epoch,
              ', mb_size = ', self.data_loader.train_batch_size)
        
        # looping over epochs
        for e in range(1, n_epoch+1):
            index = 0
            # looping over mini batches
            for i in range(1, int(np.ceil(self.data_loader.get_train_data_size() / self.data_loader.train_batch_size))+1):
                x_batch, y_batch = self.data_loader.train_data_loader(index)
                x_valid, y_valid = self.data_loader.val_data_loader(0)
                
                self.sess.run(self.train_step_tf,feed_dict={self.x_data_tf: x_batch,
                                                            self.y_data_tf: y_batch,
                                                            self.keep_prob_tf: self.keep_prob,
                                                            self.learn_rate_tf: self.learning_rate})

                feed_dict_valid = {self.x_data_tf: x_valid,
                                   self.y_data_tf: y_valid,
                                   self.keep_prob_tf: 1.0}

                feed_dict_train = {self.x_data_tf: x_batch,
                                    self.y_data_tf: y_batch,
                                    self.keep_prob_tf: 1.0}
                
                # store losses and accuracies
                if i%self.validation_step == 0:
                    valid_loss.append(sess.run(self.cross_entropy_tf,
                                               feed_dict = feed_dict_valid))
                    valid_acc.append(self.accuracy_tf.eval(session = sess, 
                                                           feed_dict = feed_dict_valid))
                    print('%.0f epoch, %.0f iteration: val loss = %.4f, val acc = %.4f'%(
                        e, i, valid_loss[-1],valid_acc[-1]))

                # summary for tensorboard
                if i%self.summary_step == 0:
                    self.n_log_step += 1 # for logging the results
                    train_summary = sess.run(self.merged, feed_dict={self.x_data_tf: x_batch, 
                                                                    self.y_data_tf: y_batch, 
                                                                    self.keep_prob_tf: 1.0
                                                                    })
                    valid_summary = sess.run(self.merged, feed_dict = feed_dict_valid)
                    self.train_writer.add_summary(train_summary, self.n_log_step)
                    self.valid_writer.add_summary(valid_summary, self.n_log_step)

                if i%self.display_step == 0:
                    train_loss.append(sess.run(self.cross_entropy_tf,
                                               feed_dict = feed_dict_train))
                    train_acc.append(self.accuracy_tf.eval(session = sess, 
                                                           feed_dict = feed_dict_train))
                    print('%.0f epoch, %.0f iteration: train loss = %.4f, train acc = %.4f'%(
                        e, i,  train_loss[-1],train_acc[-1]))

                # save current model to disk
                if i%self.checkpoint_step == 0:
                    self.save_model(sess)
                
                index += self.data_loader.train_batch_size
                
        # concatenate losses and accuracies and assign to tensor variables
        tl_c = np.concatenate([self.train_loss_tf.eval(session=sess), train_loss], axis = 0)
        vl_c = np.concatenate([self.valid_loss_tf.eval(session=sess), valid_loss], axis = 0)
        ta_c = np.concatenate([self.train_acc_tf.eval(session=sess), train_acc], axis = 0)
        va_c = np.concatenate([self.valid_acc_tf.eval(session=sess), valid_acc], axis = 0)
   
        sess.run(tf.assign(self.train_loss_tf, tl_c, validate_shape = False))
        sess.run(tf.assign(self.valid_loss_tf, vl_c , validate_shape = False))
        sess.run(tf.assign(self.train_acc_tf, ta_c , validate_shape = False))
        sess.run(tf.assign(self.valid_acc_tf, va_c , validate_shape = False))
        
        print('running time for training: ', datetime.datetime.now() - start)
        return None
    
    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        """
        Main function for model training.
        -----------------
        Parameters:
            display_step       (int)   - Number of steps we cycle through before displaying detailed progress
            validation_step    (int)   - Number of steps we cycle through before validating the model
            checkpoint_step    (int)   - Number of steps we cycle through before saving checkpoint
            summary_step       (int)   - Number of steps we cycle through before saving summary
        Returns:
            None
        -----------------
        """
        self.display_step = display_step
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.summary_step = summary_step
        
        # self.x_valid, self.y_valid = self.data_loader.all_val_data_loader()

        self.saver_tf = tf.train.Saver(max_to_keep = self.max_to_keep)

        # attach summaries
        self.attach_summary(self.sess)

        # training on original data
        self.train_model_helper(self.sess, n_epoch = self.num_epochs)

        # save final model
        self.save_model(self.sess)

        self.close_writers()

    def get_accuracy(self, sess):
        """
        Get accuracies of training and validation sets.
        -----------------
        Parameters:
            sess - session
        Returns:
            tuple (tuple of lists) train and validation accuracies
        -----------------
        """
        train_acc = self.train_acc_tf.eval(session = sess)
        valid_acc = self.valid_acc_tf.eval(session = sess)
        return train_acc, valid_acc

    def get_loss(self, sess):
        """
        Get losses of training and validation sets.
        -----------------
        Parameters:
            sess - session
        Returns:
            tuple (tuple of lists) train and validation losses
        -----------------
        """
        train_loss = self.train_loss_tf.eval(session = sess)
        valid_loss = self.valid_loss_tf.eval(session = sess)
        return train_loss, valid_loss 

    def forward(self, sess, x_data):
        """
        Forward prediction of current graph.
        Will be used in test_model method.
        -----------------
        Parameters:
            sess                 - actual session
            x_data (matrix_like) - data for which we want to calculate predicted probabilities
        Returns:
            vector_like - predicted probabilities for input data
        -----------------
        """
        y_pred_proba = self.y_pred_proba_tf.eval(session = sess, 
                                                 feed_dict = {self.x_data_tf: x_data,
                                                              self.keep_prob_tf: 1.0 })
        return y_pred_proba
    
    def load_session_from_file(self, filename):
        """
        Load session from file, restore graph, and load tensors.
        -----------------
        Parameters:
            filename (string) - the name of the model (name of file we saved in disk)
        Returns:
            session
        -----------------
        """
        tf.reset_default_graph()

        filepath = os.path.join(self.base_dir, self.model_name, 'checkpoints', filename + '.meta')
        saver = tf.train.import_meta_graph(filepath)
        sess = tf.Session()
        saver.restore(sess, os.path.join(self.base_dir, self.model_name, 'checkpoints', filename))
        graph = tf.get_default_graph()
        
        self.load_tensors(graph)
        
        return sess

    def test_model(self):
        """
        load model and test on test data.
        -----------------
        Parameters:
            None
        Returns:
            metric, defined in dnn class (for example accuracy)
        -----------------
        """
        x_test = self.data_loader.all_test_data_loader()

        sess = self.load_session_from_file(self.model_name)
        
        y_test_pred = {}
        y_test_pred_labels = {}
        y_test_pred[self.model_name] = self.forward(sess, x_test)

        sess.close()
        
        y_test_pred_labels[self.model_name] = one_hot_to_dense(y_test_pred[self.model_name])

        lbls = []
        for val in y_test_pred_labels[self.model_name]:
            if val == 0:
                lbls.append('london')
            elif val == 1:
                lbls.append('Yerevan')

        if not os.path.exists(self.inference_dir):
            os.mkdir(self.inference_dir)
        with open(os.path.join(self.inference_dir, 'inference.txt'), 'a+') as f:
            for idx in range(self.data_loader.get_test_data_size()):
                f.write("%s    :    " % self.data_loader.test_paths[idx])
                f.write("%s\n" % lbls[idx])

    def initialize_network(self):
        """
        Initialize network from last checkpoint if exists, otherwise initialize with random values.
        -----------------
        Parameters:
            None
        Returns:
            metric, defined in dnn class (for example accuracy)
        -----------------
        """
        self.sess = tf.InteractiveSession()
        filepath = os.path.join(self.base_dir, self.model_name, 'checkpoints', self.model_name + '.meta')
        if os.path.isfile(filepath) == False:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = self.load_session_from_file(self.model_name)
        return None
    
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def load_tensors(self, graph):
        raise NotImplementedError('subclasses must override metrics()!')

    @abstractmethod
    def attach_summary(self, sess):
        raise NotImplementedError('subclasses must override metrics()!')