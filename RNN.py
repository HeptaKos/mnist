import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
n_input = 28
n_steps = 28
n_hidden_unis = 128
n_classes = 10

mnists = input_data.read_data_sets("MNIST_data", one_hot=True)
x_image = tf.placeholder(tf.float32, [None, 28, 28], name="x_value")
y_value = tf.placeholder(tf.float32, [None, n_classes], name="y_value")
drop_prob = tf.placeholder(tf.float32, name="drop_prob")

def regular(parameter):
    regu = tf.contrib.layers.l2_regularizer(0.1)(parameter)
    tf.add_to_collection("loss", regu)



weights = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_unis])),
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_unis,])),
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def rnn(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x_in = tf.matmul(x, weights['in'])+biases['in']
    x_in = tf.nn.dropout(x_in, drop_prob)
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_unis])
    regular(weights['in'])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)

    result = tf.matmul(states[1], weights['out'])+biases['out']
    regular(weights['out'])
    return result


pred = rnn(x_image, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_value, logits=pred))
loss = tf.add_to_collection("loss", loss)
loss = tf.add_n(tf.get_collection("loss"))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_value, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    try:
        saver = tf.train.Saver()
        saver.restore(sess, "RNN_Model/save.ckpt")
    except:
        sess.run(tf.initialize_all_variables())
    counter = 10
    drop_value = 0.7
    for i in range(1000):
        if counter == 0:
            break
        x_batch, y_batch = mnists.train.next_batch(batch_size)
        x_batch = x_batch.reshape([batch_size, n_steps, n_input])
        sess.run(train_step, feed_dict={x_image: x_batch, y_value: y_batch, drop_prob: drop_value})
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x_image: x_batch, y_value: y_batch, drop_prob:1})
            print("The remained step is :  ", counter*100, "  accuracy is ", acc)
            counter -= 1
    tf.train.Saver().save(sess, "RNN_Model/save.ckpt")


