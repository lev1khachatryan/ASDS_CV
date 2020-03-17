from .BaseNN import *

class CNN(BaseNN):

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, inference_dir, num_epochs, train_batch_size,
                 val_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob):

        super().__init__(train_images_dir, val_images_dir, test_images_dir, inference_dir, num_epochs, train_batch_size,
                 val_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob)

        self.s_f_conv1 = 5; # filter size of first convolution layer
        self.n_f_conv1 = 32; # number of features of first convolution layer
        self.s_f_conv2 = 5; # filter size of second convolution layer
        self.n_f_conv2 = 32; # number of features of second convolution layer
        self.n_n_fc1 = 256; # number of neurons of first fully connected layer

    def weight_variable(self, shape, name = None):
        """
        Weight initialization
        -----------------
        Parameters:
            shape   (tuple)     - shape of weight variable
            name    (string)    - name of weight variable
        Returns:
            tf.Variable         - initialized weight variable
        -----------------
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name = name)

    def bias_variable(self, shape, name = None):
        """
        Bias initialization
        -----------------
        Parameters:
            shape   (tuple)     - shape of bias variable
            name    (string)    - name of bias variable
        Returns:
            tf.Variable         - initialized bias variable
        -----------------
        """
        initial = tf.constant(0.1, shape=shape) #  positive bias
        return tf.Variable(initial, name = name)

    def conv2d(self, x, W, name = None):
        """
        2D convolution
        -----------------
        Parameters:
            x       (matrix-like)   - input data
            W       (tf.Variable)   - Weight matrix
            name    (string)        - name of the graph node
        Returns:
            tf.Conv                 - Convolution response
        -----------------
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

    def max_pool_2x2(self, x, name = None):
        """
        2 x 2 max pooling
        -----------------
        Parameters:
            x       (matrix-like)   - matrix we want to apply max pooling
            name    (string)        - name of the graph node
        Returns:
            tf.max_pool             - max pooling response
        -----------------
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

    def summary_variable(self, var, var_name):
        """
        Attach summaries to a tensor for TensorBoard visualization
        -----------------
        Parameters:
            var         - variable we want to attach
            var_name    - name of the variable
        Returns:
            None
        -----------------
        """
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        return None
    
    def load_tensors(self, graph):
        """
        load tensors from a saved graph
        -----------------
        Parameters:
            graph       (tf.graph_like) - graph we obtained from saved file
        Returns:
            None
        -----------------
        """

        # input tensors
        self.x_data_tf = graph.get_tensor_by_name("x_data_tf:0")
        self.y_data_tf = graph.get_tensor_by_name("y_data_tf:0")
        
        # weights and bias tensors
        self.W_conv1_tf = graph.get_tensor_by_name("W_conv1_tf:0")
        self.W_conv2_tf = graph.get_tensor_by_name("W_conv2_tf:0")
        self.W_fc1_tf = graph.get_tensor_by_name("W_fc1_tf:0")
        self.W_fc2_tf = graph.get_tensor_by_name("W_fc2_tf:0")
        self.b_conv1_tf = graph.get_tensor_by_name("b_conv1_tf:0")
        self.b_conv2_tf = graph.get_tensor_by_name("b_conv2_tf:0")
        self.b_fc1_tf = graph.get_tensor_by_name("b_fc1_tf:0")
        self.b_fc2_tf = graph.get_tensor_by_name("b_fc2_tf:0")
        
        # activation tensors
        self.h_conv1_tf = graph.get_tensor_by_name('h_conv1_tf:0')
        self.h_pool1_tf = graph.get_tensor_by_name('h_pool1_tf:0')
        self.h_conv2_tf = graph.get_tensor_by_name('h_conv2_tf:0')
        self.h_pool2_tf = graph.get_tensor_by_name('h_pool2_tf:0')
        self.h_fc1_tf = graph.get_tensor_by_name('h_fc1_tf:0')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        
        # training and prediction tensors
        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
        self.cross_entropy_tf = graph.get_tensor_by_name('cross_entropy_tf:0')
        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        self.y_pred_proba_tf = graph.get_tensor_by_name("y_pred_proba_tf:0")
        self.y_pred_correct_tf = graph.get_tensor_by_name('y_pred_correct_tf:0')
        self.accuracy_tf = graph.get_tensor_by_name('accuracy_tf:0')
        
        # tensor of stored losses and accuracies during training
        self.train_loss_tf = graph.get_tensor_by_name("train_loss_tf:0")
        self.train_acc_tf = graph.get_tensor_by_name("train_acc_tf:0")
        self.valid_loss_tf = graph.get_tensor_by_name("valid_loss_tf:0")
        self.valid_acc_tf = graph.get_tensor_by_name("valid_acc_tf:0")

        return None

    def attach_summary(self, sess):
        """
        Create summary tensors for tensorboard.
        -----------------
        Parameters:
            sess - the session for which we want to create summaries
        Returns:
            None
        -----------------
        """
        self.summary_variable(self.W_conv1_tf, 'W_conv1_tf')
        self.summary_variable(self.b_conv1_tf, 'b_conv1_tf')
        self.summary_variable(self.W_conv2_tf, 'W_conv2_tf')
        self.summary_variable(self.b_conv2_tf, 'b_conv2_tf')
        self.summary_variable(self.W_fc1_tf, 'W_fc1_tf')
        self.summary_variable(self.b_fc1_tf, 'b_fc1_tf')
        self.summary_variable(self.W_fc2_tf, 'W_fc2_tf')
        self.summary_variable(self.b_fc2_tf, 'b_fc2_tf')
        tf.summary.scalar('cross_entropy_tf', self.cross_entropy_tf)
        tf.summary.scalar('accuracy_tf', self.accuracy_tf)

        # merge all summaries for tensorboard
        self.merged = tf.summary.merge_all()

        # initialize summary writer 
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(self.base_dir, self.model_name, 'logs', (self.model_name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

        return None

    def network(self, X):
        """
        Construct network architecture
        -----------------
        Parameters:
            X       (tensor) - input data with shape of (batch size; height of image; width of image ; num channels)
        Returns:
            Last layer of the network (for continuation)
        -----------------
        """

        # 1.layer: convolution + max pooling
        self.W_conv1_tf = self.weight_variable([self.s_f_conv1, self.s_f_conv1, self.data_loader.num_channels, self.n_f_conv1], name = 'W_conv1_tf') # (5,5,3,32)
        self.b_conv1_tf = self.bias_variable([self.n_f_conv1], name = 'b_conv1_tf') # (32)
        self.h_conv1_tf = tf.nn.relu(self.conv2d(X, self.W_conv1_tf) + self.b_conv1_tf, name = 'h_conv1_tf') # (.,100,100,32)
        self.h_pool1_tf = self.max_pool_2x2(self.h_conv1_tf, name = 'h_pool1_tf') # (.,50,50,32)

        # 2.layer: convolution + max pooling
        self.W_conv2_tf = self.weight_variable([self.s_f_conv2, self.s_f_conv2, self.n_f_conv1, self.n_f_conv2], name = 'W_conv2_tf') # (5,5,32,32)
        self.b_conv2_tf = self.bias_variable([self.n_f_conv2], name = 'b_conv2_tf') # (32)
        self.h_conv2_tf = tf.nn.relu(self.conv2d(self.h_pool1_tf, self.W_conv2_tf) + self.b_conv2_tf, name ='h_conv2_tf') #(.,50,50,32)
        self.h_pool2_tf = self.max_pool_2x2(self.h_conv2_tf, name = 'h_pool2_tf') #(.,25,25,32)

        # 3.layer: fully connected
        self.W_fc1_tf = self.weight_variable([25*25*self.n_f_conv2,self.n_n_fc1], name = 'W_fc1_tf') # (25*25*32=20000, 256)
        self.b_fc1_tf = self.bias_variable([self.n_n_fc1], name = 'b_fc1_tf') # (256)
        self.h_pool2_flat_tf = tf.reshape(self.h_pool2_tf, [-1,25*25*self.n_f_conv2], name = 'h_pool3_flat_tf') # (., 20000)
        self.h_fc1_tf = tf.nn.relu(tf.matmul(self.h_pool2_flat_tf, self.W_fc1_tf) + self.b_fc1_tf, name = 'h_fc1_tf') # (.,256)
      
        # add dropout
        self.keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
        self.h_fc1_drop_tf = tf.nn.dropout(self.h_fc1_tf, self.keep_prob_tf, name = 'h_fc1_drop_tf')

        # 4.layer: fully connected
        self.W_fc2_tf = self.weight_variable([self.n_n_fc1, self.data_loader.num_classes], name = 'W_fc2_tf')
        self.b_fc2_tf = self.bias_variable([self.data_loader.num_classes], name = 'b_fc2_tf')
        
        self.z_pred_tf = tf.add(tf.matmul(self.h_fc1_drop_tf, self.W_fc2_tf), self.b_fc2_tf, name = 'z_pred_tf')# => (.,2)

        return self.z_pred_tf