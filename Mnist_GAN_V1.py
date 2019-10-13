import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

mnist = input_data.read_data_sets("MNIST_data", one_hot="True")


def init_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def init_biases(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


x1 = tf.placeholder(tf.float32, shape=[None, 100], name="Noise")
x2 = tf.placeholder(tf.float32, shape=[None, 28*28], name="Noise")
x2_image = tf.reshape(x2, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)


def generator(x):
    w1 = init_weight([100, 7*7*256])
    b1 = init_biases([7*7*256])
    h1 = tf.nn.leaky_relu((tf.matmul(x, w1)+b1))
    h1 = tf.reshape(h1, [-1, 7, 7, 256])

    w2 = init_weight([5, 5, 128, 256])
    b2 = init_biases([128])
    h2 = tf.nn.leaky_relu(tf.nn.conv2d_transpose(h1, w2, strides=[1, 1, 1, 1], padding="SAME",
                                                 output_shape=[100, 7, 7, 128])+b2) # 7*7*128

    w3 = init_weight([5, 5, 64, 128])
    b3 = init_biases([64])
    h3 = tf.nn.leaky_relu(tf.nn.conv2d_transpose(h2, w3, strides=[1, 2, 2, 1], padding="SAME",
                                                 output_shape=[100, 14, 14, 64])+b3)  # 14*14*64

    w4 = init_weight([5, 5, 1, 64])
    h4 = tf.nn.leaky_relu(tf.nn.conv2d_transpose(h3, w4, strides=[1, 2, 2, 1], padding="SAME",
                                                 output_shape=[100, 28, 28, 1]))  # 28*28*1

    return h4


def discriminator(true_x, fake_x):
    x = tf.concat([true_x, fake_x], 0)

    w1 = init_weight([5, 5, 1, 64])
    b1 = init_biases([64])
    h1 = tf.nn.leaky_relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="SAME")+b1)  # 28*28*64
    h1 = tf.nn.dropout(h1, keep_prob)

    h1 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # 14*14*64

    w2 = init_weight([5, 5, 64, 128])
    b2 = init_biases([128])
    h2 = tf.nn.leaky_relu(tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding="SAME") + b2)  # 14*14*128
    h2 = tf.nn.dropout(h2, keep_prob)

    h2 = tf.nn.max_pool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # 7*7*128

    w3 = init_weight([5, 5, 128, 256])
    b3 = init_biases([256])
    h3 = tf.nn.leaky_relu(tf.nn.conv2d(h2, w3, strides=[1, 1, 1, 1], padding="SAME")+b3)  # 7*7*256
    h3 = tf.nn.dropout(h3, keep_prob)
    print(h3.shape)

    h3 = tf.reshape(h3, [-1, 7*7*256])

    w4 = init_weight([7*7*256, 1])
    h4 = tf.nn.tanh(tf.matmul(h3, w4))

    true_out = tf.slice(h4, [0, 0], [BATCH_SIZE, -1], name="TRUE_OUT")
    fake_out = tf.slice(h4, [BATCH_SIZE+1, 0], [-1, -1], name="FAKE_OUT")
    return true_out, fake_out


def generator_loss(fake_out):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones_like(fake_out), logits=fake_out))


def discriminator_loss(fake_out, true_out):
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.zeros_like(fake_out), logits=fake_out))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones_like(true_out), logits=true_out))
    return loss1+loss2

Generate_Image = generator(x1)
True_out, Fake_out = discriminator(x2_image, Generate_Image)
loss = generator_loss(Fake_out) + discriminator_loss(Fake_out, True_out)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    counter = 10
    for i in range(100):
        if i % 10 == 0:
            counter = counter - 1
            print("remained step is :  ", counter * 10)
        batch_x, _ = mnist.train.next_batch(100)
        noises = np.random.randn(100, 100)
        sess.run(train_step, feed_dict={x1: noises, x2: batch_x, keep_prob:0.5})







