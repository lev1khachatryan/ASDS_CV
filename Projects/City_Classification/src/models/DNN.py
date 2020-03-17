from .BaseNN import *

class DNN(BaseNN):

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, inference_dir, num_epochs, train_batch_size,
                 val_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob):

        super().__init__(train_images_dir, val_images_dir, test_images_dir, inference_dir, num_epochs, train_batch_size,
                 val_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob)

        self.n_n_fc1 = 256
        self.n_n_fc2 = 64


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
        self.W_fc1_tf = graph.get_tensor_by_name("W_fc1_tf:0")
        self.W_fc2_tf = graph.get_tensor_by_name("W_fc2_tf:0")
        self.W_fc3_tf = graph.get_tensor_by_name("W_fc3_tf:0")
        self.b_fc1_tf = graph.get_tensor_by_name("b_fc1_tf:0")
        self.b_fc2_tf = graph.get_tensor_by_name("b_fc2_tf:0")
        self.b_fc3_tf = graph.get_tensor_by_name("b_fc3_tf:0")
        
        # activation tensors
        self.h_fc1_tf = graph.get_tensor_by_name('h_fc1_tf:0')
        self.h_fc2_tf = graph.get_tensor_by_name('h_fc2_tf:0')
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
        self.summary_variable(self.W_fc1_tf, 'W_fc1_tf')
        self.summary_variable(self.b_fc1_tf, 'b_fc1_tf')
        self.summary_variable(self.W_fc2_tf, 'W_fc2_tf')
        self.summary_variable(self.b_fc2_tf, 'b_fc2_tf')
        self.summary_variable(self.W_fc3_tf, 'W_fc3_tf')
        self.summary_variable(self.b_fc3_tf, 'b_fc3_tf')
        tf.summary.scalar('cross_entropy_tf', self.cross_entropy_tf)
        tf.summary.scalar('accuracy_tf', self.accuracy_tf)

        # merge all summaries for tensorboard
        self.merged = tf.summary.merge_all()

        # initialize summary writer 
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        # filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'logs', (self.model_name+'_'+timestamp))
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
        # 1.layer
        self.W_fc1_tf = self.weight_variable([self.data_loader.height_of_image*self.data_loader.width_of_image*self.data_loader.num_channels, self.n_n_fc1], name = 'W_fc1_tf') # (100*100*3=30000, 256)
        self.b_fc1_tf = self.bias_variable([self.n_n_fc1], name = 'b_fc1_tf') # (256)
        self.x_flat_tf = tf.reshape(X, [-1,self.data_loader.height_of_image*self.data_loader.width_of_image*self.data_loader.num_channels], name = 'x_flat_tf') # (., 30000)
        self.h_fc1_tf = tf.nn.relu(tf.matmul(self.x_flat_tf, self.W_fc1_tf) + self.b_fc1_tf, name = 'h_fc1_tf') # (.,256)
      
        # add dropout
        self.keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
        self.h_fc1_drop_tf = tf.nn.dropout(self.h_fc1_tf, self.keep_prob_tf, name = 'h_fc1_drop_tf')

        # 2.layer
        self.W_fc2_tf = self.weight_variable([self.n_n_fc1, self.n_n_fc2], name = 'W_fc2_tf') # (256, 64)
        self.b_fc2_tf = self.bias_variable([self.n_n_fc2], name = 'b_fc2_tf') # (64)
        self.h_fc2_tf = tf.nn.relu(tf.matmul(self.h_fc1_drop_tf, self.W_fc2_tf) + self.b_fc2_tf, name = 'h_fc2_tf') # (.,64)
      
        # add dropout
        self.h_fc2_drop_tf = tf.nn.dropout(self.h_fc2_tf, self.keep_prob_tf, name = 'h_fc2_drop_tf')

        # 3.layer: fully connected
        self.W_fc3_tf = self.weight_variable([self.n_n_fc2, self.data_loader.num_classes], name = 'W_fc3_tf') # (64, 2)
        self.b_fc3_tf = self.bias_variable([self.data_loader.num_classes], name = 'b_fc3_tf') # (2)
        
        self.z_pred_tf = tf.add(tf.matmul(self.h_fc2_drop_tf, self.W_fc3_tf), self.b_fc3_tf, name = 'z_pred_tf')# => (.,2)

        return self.z_pred_tf