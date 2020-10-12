from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np

print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Let start our work: creating a convolutional neural network

#####TO COMPLETE

def processed_data(x_train, x_test):
    """
    - Loads cifar 10 data
    - Normalize the data
    - change it to flot for computational reasons
    """
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, x_test


def visualize_sample_images(class_names, x_train, y_train):

    plt.figure(figsize=(10, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[y_train[i][0]])
    plt.show()


if __name__ == '__main__':

    # load dataset from the keras api
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # visualize sample images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # download and preprocess image
    x_train, x_test = processed_data(x_train, x_test)

    # visualize sample images
    visualize_sample_images(class_names, x_train, y_train)
