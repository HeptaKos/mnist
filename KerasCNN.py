import tensorflow as tf
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)
print(train_images.shape)
train_images = tf.to_float(train_images)/255
test_images = tf.to_float(test_images)/255

train_labels = tf.one_hot(tf.convert_to_tensor(train_labels), 10, 1, 0)
test_labels = tf.one_hot(tf.convert_to_tensor(test_labels), 10, 1, 0)
print(train_labels.shape)

def new_model():
    model = keras.Sequential([
        keras.layers.Conv2D(8, [3, 3], [1, 1], padding="SAME", activation="relu"),
        keras.layers.MaxPooling2D([2, 2], [2, 2], padding="VALID"),
        keras.layers.Flatten(input_shape=(14, 14, 8)),
        keras.layers.Dense(256, use_bias=True, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model():
    model = keras.models.load_model("KerasCNN_Model/model.h5")
    model.summary()
    return model
try:
    Model = get_model()
except:
    Model = new_model()

Model.fit(train_images, train_labels, steps_per_epoch=100, batch_size=1000)

test_loss, test_acc = Model.evaluate(test_images, test_labels, steps=1)
print("Test acc is :  ", test_acc)

Model.save("KerasCNN_Model/model.h5")
