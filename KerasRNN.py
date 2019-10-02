import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.examples.tutorials.mnist import input_data

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

def new_model():
    model = keras.Sequential([
        keras.layers.LSTM(40, input_shape=(train_images.shape[1], train_images.shape[2])),
        keras.layers.Dense(128, activation="relu", use_bias=True),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def get_model():
    model = keras.models.load_model("KerasRNN_Model/model.h5")
    model.summary()
    return model

print(train_images.shape)
train_images = tf.to_float(train_images)/255
test_images = tf.to_float(test_images)/255

train_labels = tf.one_hot(tf.convert_to_tensor(train_labels), 10, 1, 0)
test_labels = tf.one_hot(tf.convert_to_tensor(test_labels), 10, 1, 0)
train_labels = tf.to_int32(train_labels)
test_labels = tf.to_int32(test_labels)

try:
    Model = get_model()
except:
    Model = new_model()

Model.fit(train_images, train_labels, steps_per_epoch=10, batch_size=1000)
test_loss, test_acc = Model.evaluate(test_images, test_labels, steps=1)
print("Test acc is :  ", test_acc)

Model.save("KerasRNN_Model/model.h5")

