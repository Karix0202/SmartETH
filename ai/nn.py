import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

tf.reset_default_graph()

class EthernumPriceNeuralNetwork:
    def __init__(
        self, 
        n_inputs=1, 
        n_outputs=1,
        n_neurons=200,
        n_iterations=200,
        n_time_steps=1,
        learning_rate=.03,
        n_layers=800
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.n_iterations = n_iterations
        self.n_time_steps = n_time_steps
        self.learning_rate = learning_rate
        self.n_layers = n_layers

        self.x = tf.placeholder(tf.float32, [None, self.n_time_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_time_steps, self.n_outputs])

    def next_batch(self, data):
        rand_start = np.random.randint(len(data) - self.n_time_steps)
        y_batch = np.array(data[rand_start:rand_start + self.n_time_steps + 1]).reshape(1, self.n_time_steps + 1)
        
        return y_batch[: ,:-1].reshape(-1, self.n_time_steps, 1),y_batch[:, 1:].reshape(-1, self.n_time_steps, 1)
    
    def create_graph(self):
        lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=tf.nn.relu), output_size=self.n_outputs)
        return tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float32)

    def train(self, data, model_name='', print_mse=True):
        outputs, _ = self.create_graph()

        loss = tf.reduce_mean(tf.square(outputs - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)    
    
            for i in range(self.n_iterations):    
                x_batch, y_batch = self.next_batch(data)

                sess.run(optimizer, feed_dict={
                    self.x:x_batch,
                    self.y:y_batch
                })

                if print_mse:
                    if i % 10 == 0:
                        mse = loss.eval(feed_dict={
                            self.x: x_batch,
                            self.y: y_batch
                        })
                        print('I: {}, MSE: {}'.format(i, mse))

                if model_name != '':
                    saver.save(sess, model_name)