import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

def Conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')


def MaxPooling(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def init_weight(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)

def get_accuracy(set, labels):
    global full_conn2_out
    y_prediction = sess.run(full_conn2_out, feed_dict={x_value: set, drop_prob: 1})
    correct_classified = tf.equal(tf.argmax(labels, 1), tf.argmax(y_prediction, 1))
    compute_acc = tf.reduce_mean(tf.cast(correct_classified, tf.float32))
    result = sess.run(compute_acc)
    return result


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_value = tf.placeholder(tf.float32, [None, 28 * 28], name="x_value")
y_value = tf.placeholder(tf.float32, [None, 10], name="y_value")
drop_prob = tf.placeholder(tf.float32, name="drop_prob")

x_image = tf.reshape(x_value, [-1, 28, 28, 1])

conv1_w = init_weight([5, 5, 1, 32])
conv1_b = init_bias([32])
conv1 = tf.nn.relu(Conv2d(x_image,conv1_w)+conv1_b)
pool1 = MaxPooling(conv1)  #14 14 1 32

conv2_w = init_weight([5, 5, 32, 64])
conv2_b = init_bias([64])
conv2 = tf.nn.relu(Conv2d(pool1, conv2_w)+conv2_b)
pool2 = MaxPooling(conv2)  #7 7 1 64

full_conn1_in = tf.reshape(pool2, [-1, 7*7*64])
full_conn1_w = init_weight([7*7*64, 1024*2])
full_conn1_b = init_bias([1024*2])
full_conn1_out = tf.nn.relu(tf.matmul(full_conn1_in, full_conn1_w)+full_conn1_b)
full_conn1_drop = tf.nn.dropout(full_conn1_out, keep_prob=drop_prob)

full_conn2_w = init_weight([1024*2, 10])
full_conn2_b = init_bias([10])
full_conn2_out = tf.nn.softmax(tf.matmul(full_conn1_drop, full_conn2_w)+full_conn2_b)

loss = tf.reduce_mean(-tf.reduce_sum(y_value*tf.log(full_conn2_out), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    saver = tf.train.Saver()                #加载模型操作不能定义在函数中
    saver.restore(sess, "CNN_Model/save.ckpt")
    counter = 10
    for step in range(1000):
        if(counter == 0):
            break
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x_value: batch_x, y_value: batch_y, drop_prob: 1 })
        if step % 100 == 0:
            counter -= 1
            print("remained steps are: ", counter*100)
            print("steps accuracy is : ", get_accuracy(mnist.test.images, mnist.test.labels))


    saver = tf.train.Saver()
    save_path = saver.save(sess, 'CNN_Model/save.ckpt')

    print('Training DONE!')
