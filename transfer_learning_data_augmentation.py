from __future__ import print_function

# The two folloing lines allow to reduce tensorflow verbosity
import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '1'  # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING
# msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.applications import vgg16

import matplotlib.pyplot as plt
import numpy as np

print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


# load and prepocess data
def load_and_process_data():
    """
    This model loads the CIFAR 10 dataset from the keras api

    """
    num_classes = 10

    # load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    num_classes = 10
    print('y_train.shape=', y_train.shape)
    print('y_test.shape=', y_test.shape)

    # Convert class vectors to binary class matrices ("one hot encoding")
    ## Doc : https://keras.io/utils/#to_categorical
    y_train = tensorflow.keras.utils.to_categorical(y_train)
    y_test = tensorflow.keras.utils.to_categorical(y_test)

    # Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train.shape=', x_train.shape)
    print('x_test.shape=', x_test.shape)

    # Normalize inputs from [0; 255] to [0; 1]
    x_train = x_train / 255
    x_test = x_test / 255
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8)

    return x_train, y_train, x_test, y_test, x_val, y_val


# def data agumentation
def augment_data(x_train, y_train, x_val, y_val, batch_size=128):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        horizontal_flip=True
     )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        horizontal_flip=True)

    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    val_datagen.fit(x_val)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

    return train_generator, val_generator


# draw the training statistics
def plot_training_metrics(choice, history):
    # directory to save the plots
    path = os.path.join('results/cifar10', str(choice))
    if not os.path.exists(path):
        os.makedirs(path)
    # plotting the metrics
    fig = plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    #plt.savefig('{}/cifar-accuracy-model-{}.png'.format(path, choice), format='png')
    plt.show()

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.tight_layout()
    #plt.savefig('{}/cifar-loss-model-{}.png'.format(path, choice), format='png')
    plt.show()

# save model
def save_model(model, model_name):
    # saving the model
    save_dir = "results/cifar10/" + str(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = str(model_name) + '.h5'
    model_path = os.path.join(save_dir, file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def Network(lr,retrain_vgg=False):
    # get VGG16
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    if (retrain_vgg==False):
        for layer in base_model.layers:
            layer.trainable = False


    # build New Network
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # set optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr),
                  metrics=['accuracy'])

    return model


# Function to Run the model
def run_program():


    ####################### Hyper-parameters##################################
    
    choice = 1
    retrain_vgg=False
    epochs = 5
    batch=128
    data_augment=False
    learning_rate=0.0001
    ########################################################################
    
    #Define the model
    model = Network(learning_rate,retrain_vgg=retrain_vgg)

    # Get the model summary
    model.summary()
    
    # get data
    x_train, y_train, x_test, y_test, x_val, y_val = load_and_process_data()

    # agument data & train the model
    if (data_augment==True): 
        train_augmented, val_agumented = augment_data(x_train, y_train, x_val, y_val)
        hist = model.fit_generator(train_augmented,
                        validation_data=val_agumented,
                        epochs=epochs,
                        verbose=1)
    else:
        # train model
        hist = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_data=(x_val, y_val))



    # display training details
    plot_training_metrics(choice, hist)

    # evaluate model
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss: ", loss)
    print("Test Accuracy", acc)

    #save model
    #save_model(model, choice)


if __name__ == "__main__":
    # ran everthing from here
    run_program()

