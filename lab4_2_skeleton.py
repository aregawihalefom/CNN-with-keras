from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

import matplotlib.pyplot as plt
import numpy as np
import os

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


def model1():
    # creating model instance
    model1 = Sequential()

    # first Convolutuional layers
    model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))

    # Maxpooling Layers
    model1.add(MaxPooling2D(2, 2))

    # Flatten Layers
    model1.add(Flatten())

    # Linear Layer
    model1.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

    # Final Layer with softmax activation
    model1.add(Dense(10, activation='softmax'))

    # complile model
    optm = SGD(lr=0.001, momentum=0.9)
    model1.compile(optimizer=optm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model1


def Model3():
    model3 = Sequential()
    model3.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',
                      input_shape=(32, 32, 3)))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Flatten())
    model3.add(Dense(256, activation='relu'))
    model3.add(Dense(10, activation='softmax'))

    optm = Adam(learning_rate=0.01)
    model3.compile(optimizer=optm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model3


def train_model(model, x_train, y_train, batch_size, epochs, validation_split):
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    return hist


def plot_training_metrics(choosen_model, history):
    # directory to save the plots
    path = os.path.join('results', str(choosen_model))
    if not os.path.exists(path):
        os.mkdir(path)
    # plotting the metrics
    fig = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('{}/accuracy-model-{}.png'.format(path, choosen_model), format='png')

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.tight_layout()
    plt.savefig('{}/loss-model-{}.png'.format(path, choosen_model), format='png')


def evaluate_model(model, X_test, Y_test):
    evaluation_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", evaluation_metrics[0])
    print("Test Accuracy", evaluation_metrics[1])

    return evaluation_metrics


def visualize_correct_and_wrong_predictions(choosen_model, model, X_test, y_test):
    path = os.path.join('results', str(choosen_model))
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    predicted_classes = model.predict_classes(X_test)
    y_reversed = np.argmax(y_test, axis=1)
    # see which we predicted correctly and which not
    correct_indices = np.nonzero(predicted_classes == y_reversed)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_reversed)[0]
    print()
    print(correct_indices.shape[0], " classified correctly")
    print(incorrect_indices.shape[0], " classified incorrectly")

    # adapt figure size to accomodate 18 subplots

    plt.figure(3)
    wrong_nine = incorrect_indices[:18]
    # plot 9 incorrect predictions
    for i, incorrect in enumerate(wrong_nine):
        plt.subplot(6, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title(
            "Predicted = {}".format(predicted_classes[incorrect]))
        plt.xticks([])
        plt.yticks([])

    plt.savefig('{}/wrongly-classified-model-{}.png'.format(path, choosen_model), format='png')
    plt.tight_layout()


def save_model(model, model_name):
    # saving the model
    save_dir = "results/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_name = model_name + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    # load dataset from the keras api
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # visualize sample images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # download and preprocess image
    x_train, x_test = processed_data(x_train, x_test)

    # visualize sample images
    #visualize_sample_images(class_names, x_train, y_train)

    choose_model = 3
    # crate the model
    if choose_model == 1:
        model = model1()
        model_name = 'cifar-cnn-1'

    elif choose_model == 2:
        # model = Model2('Adam', lr=0.001)
        model_name = 'cifar-cnn-2'
    elif choose_model == 3:
        model = Model3()
        model_name = 'cifar-cnn-3'

    # train the model
    # Hyper parameters
    batch_size = 128
    epochs = 3
    validation_split = 0.2
    history = train_model(model, x_train, y_train, batch_size=batch_size, epochs=epochs,
                          validation_split=validation_split)

    # visualize training statistics
    plot_training_metrics(choose_model, history)

    # evaluate the model
    evaluation_metrics = evaluate_model(model, x_test, y_test)

    # save the model
    save_model(model, model_name)

    plt.show()
