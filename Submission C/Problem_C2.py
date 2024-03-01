# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):
            self.model.stop_training = True


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # NORMALIZE YOUR IMAGE HERE
    (images_train, labels_train), (images_test, labels_test) = mnist.load_data()

    images_train = images_train.reshape((images_train.shape[0], 28, 28, 1)) / 255.0
    images_test = images_test.reshape((images_test.shape[0], 28, 28, 1)) / 255.0

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')

    ])

    # COMPILE MODEL HERE
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    callbacks = myCallback()

    model.fit(images_train, labels_train, epochs=100, validation_data=(images_test, labels_test), callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
