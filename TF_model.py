# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:35:26 2017

@author: snljdk
"""
import tensorflow as tf
import os

class CNN_model():
    """
    This object will contain the Convolutional Neural Network.
    """
    def __init__(self, learning_rate=0.0001):
        # Specify the parameters.
        print("Building network...")
        self.graph = tf.Graph()
        
        self.lr = learning_rate
        self.update_counter = 0

        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                updates = tf.train.RMSPropOptimizer(self.lr)
            with tf.variable_scope('input'):
                # These variables represent the variables that go into the network
                self.input = tf.placeholder(tf.float32, [None, 180, 180, 3], name='input')
                self.label = tf.placeholder(tf.float32, [None], name='label')
                
        # Create the network
        logits = self.inference(self.input)
        self.preds = tf.argmax(logits, name='prediction')
        # Calculate the loss
        loss = self.loss_function(logits, self.label)
        
        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                # Only update the variables of the action network
                self.opt_operation = updates.minimize(loss)
                
        self.sess = tf.Session(graph=self.graph)
        print("Finished building network.")         
        
    def train(self, batch, summarize=False):
            """
            Train one batch.
    
            Keyword arguments:
            states -- mini batch of states
            actions -- mini batch of actions
            rewards -- mini batch of rewards
            next_states -- mini batch of next states
    
            Returns: average squared loss
            """
            images, labels = batch
    
            var_dict = {self.input:images, self.label:labels}
    
            # Calculate action network outputs
            if (self.update_counter % 1000 == 0):
                # Only write the summaries every 1000 steps.
                summary, _ = self.sess.run([self.summaries, self.opt_operation], feed_dict=var_dict)
                self.train_writer.add_summary(summary, self.update_counter)
            else:
                self.sess.run(self.opt_operation, feed_dict=var_dict)
            self.update_counter += 1
        
    def inference(self, x):
        """
            This method created the architecture of the model.
            Input:
                - Input matrix x of size [batch size x 180 x 180 x 3]
            Output:
                - The logit values for each category [batch size x 5000]
        """
        init_func = tf.contrib.layers.xavier_initializer()
        reg = tf.contrib.layers.l2_regularizer()
        
        with self.graph.as_default():
            with tf.variable_scope('conv1'):
                conv1_w = tf.get_variable('filter', (8, 8, 3, 32), initializer=init_func, regularizer=reg)
                conv1_b = tf.get_variable('biases', (32,), initializer=tf.constant_initializer(0.1))
                # Do the calculations of this layer
                h = tf.nn.conv2d(x, conv1_w, strides=[1, 4, 4, 1], padding='VALID')
                h = tf.nn.bias_add(h, conv1_b)
                h = tf.nn.relu(h, name='activation')
            with tf.variable_scope('flatten'):
                # Reshape the output of the conv layers to be flat
                n_input = h.get_shape().as_list()[1] * h.get_shape().as_list()[2] * h.get_shape().as_list()[3]
                h = tf.reshape(h, [-1, n_input])
            with tf.variable_scope('lin1'):
                lin1_w = tf.get_variable('weights', (n_input, 512), initializer=init_func, regularizer=reg)
                lin1_b = tf.get_variable('biases', (512,), initializer=tf.constant_initializer(0.1))
                h = tf.matmul(h, lin1_w) + lin1_b
            with tf.variable_scope('lin2'):
                lin2_w = tf.get_variable('weights', (512, 5000), initializer=init_func, regularizer=reg)
                lin2_b = tf.get_variable('biases', (5000,), initializer=tf.constant_initializer(0.1))
                logits = tf.matmul(h, lin2_w) + lin2_b

        return logits
    
    def loss(self, logits, labels):
        """
            This is the loss function of the model.
            Input:
                - The logits for each category [batch size x 5000]
                - The actual labels for each data point [batch size x 1]
        """
        with self.graph.as_default():
            with tf.variable_scope('loss'):
                onehot_labels = tf.one_hot(labels, 5000);
                
                losses = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels)
                loss = tf.reduce_mean(losses)
                tf.summary.scalar('cross_entropy', loss)
                
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss = tf.reduce_sum(reg_losses)
                tf.summary.scalar('regularization_loss', reg_loss)
        
        return loss + reg_loss
        
    def store(self, path, time_step):
        """
            Retrieve network weights for action and target networks
            and store these in separate files.
        """
        with self.graph.as_default():
            file_name = os.path.join(path, "network")
            print("Saving networks...")
            self.saver.save(self.sess, file_name, time_step)
            print("Saved!")
        
    def load(self, path, nr_of_saves, iteration=-1):
        """
            Load network weights for action and target networks 
            and load these into separate networks.
        """
        with self.graph.as_default():
            print("Loading networks...")
            checkpoint_dir = os.path.join(path, "network-"+str(iteration))
            self.saver = tf.train.Saver(max_to_keep=nr_of_saves+1)
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                print("Loaded: {}".format(checkpoint_dir))
            except tf.errors.InvalidArgumentError:
                if iteration <= 0:
                    # Initialize the variables
                    print("Failed! Initializing the network variables...")
                    self.sess.run(tf.global_variables_initializer())
                else:
                    raise
            
            summary_path = os.path.join("results", path)
            self.train_writer = tf.summary.FileWriter(os.path.join(summary_path, "train"), self.sess.graph)
            self.test_writer = tf.summary.FileWriter(os.path.join(summary_path, "test"))
            self.summaries = tf.summary.merge_all()