import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


NOISE_SIZE = 100
PICTURE_WIDTH = 28
PICTURE_HEIGHT = 28
LEARNING_RATE = 1e-4
BATCH_SIZE = 250
EPOCH = 10
UNITS = 256

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def gen(noise):
    with tf.variable_scope('gen'):
        h1 = tf.layers.dense(inputs=noise, units=UNITS)
        reLu = tf.nn.leaky_relu(h1, 0.01)
        drop = tf.layers.dropout(reLu, rate=0.2)
        h2 = tf.layers.dense(inputs=drop, units=UNITS)
        reLu2 = tf.nn.leaky_relu(h2, 0.01)
        drop2 = tf.layers.dropout(reLu2, rate=0.2)
        h3 = tf.layers.dense(inputs=drop2, units=PICTURE_WIDTH*PICTURE_HEIGHT)
        h4 = tf.tanh(h3)
        return h4, h3


def dis(image):
    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
        h5 = tf.layers.dense(inputs=image, units=UNITS)
        reLu3 = tf.nn.leaky_relu(h5, 0.01)
        drop3 = tf.layers.dropout(reLu3, rate=0.2)
        h7 = tf.layers.dense(inputs=drop3, units=1)
        h8 = tf.sigmoid(h7)
        return h8, h7


def loss_value(true_out, fake_out):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out, labels=tf.ones_like(fake_out)))
    t_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out, labels=tf.zeros_like(fake_out)))
    f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=true_out, labels=tf.ones_like(true_out)))
    d_loss = tf.add(t_loss, f_loss)
    return tf.add(g_loss,g_loss), d_loss


def optimizer(g_loss, d_loss):
    op_vars = tf.trainable_variables()
    g_var = [var for var in op_vars if var.name.startswith('gen')]
    d_var = [var for var in op_vars if var.name.startswith('dis')]
    g_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_var)
    d_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_var)

    return g_optimizer, d_optimizer


def train():
    image_value = tf.placeholder(tf.float32, [None, PICTURE_HEIGHT * PICTURE_WIDTH], name="x_value")
    noise_value = tf.placeholder(tf.float32, [None, NOISE_SIZE], name="x_value")
    true_image = image_value
    fake_image, logits1 = gen(noise_value)
    true_out, logits2 = dis(true_image)
    fake_out, logits3 = dis(fake_image)
    g_loss, d_loss = loss_value(logits2, logits3)
    g_optimizer, d_optimizer = optimizer(g_loss, d_loss)

    with tf.Session() as sess:
        saver = tf.train.Saver()  # 加载模型操作不能定义在函数中
        saver.restore(sess, "GAN_Model/save.ckpt")
        for epoch in range(EPOCH):
            for i in range(60000 // BATCH_SIZE):
                train_image, _ = mnist.train.next_batch(BATCH_SIZE)
                true_image = (true_image-127.5)/127.5
                noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_SIZE))
                sess.run(g_optimizer, feed_dict={noise_value: noise})
                if i%1 == 0:
                    sess.run(d_optimizer, feed_dict={noise_value: noise, image_value: train_image})
            step_g_loss = sess.run(g_loss, feed_dict={noise_value: noise})
            step_d_loss = sess.run(d_loss, feed_dict={noise_value: noise, image_value: train_image})
            print("g_loss : ", step_g_loss, "  d_loss :  ", step_d_loss)
            gen_image = sess.run(fake_image, feed_dict={noise_value: noise})
            gen_image = tf.reshape(gen_image, [-1, 28, 28])
            gen_image = gen_image.eval()
            print(gen_image.shape)
            plt.imshow(gen_image[0,: ,:], cmap='gray')
            plt.show()
        saver = tf.train.Saver()
        save_path = saver.save(sess, 'GAN_Model/save.ckpt')





if __name__ == '__main__':
    train()


