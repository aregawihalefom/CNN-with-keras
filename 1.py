from __future__ import print_function

# The two folloing lines allow to reduce tensorflow verbosity
import os

from sklearn.model_selection import train_test_split

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '1'  # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING
# msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

import matplotlib.pyplot as plt
import numpy as np

print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


# load and prepocess data
def load_and_process_data():
    """
    This model loads the MNIST dataset from the keras api

    """
    num_classes = 10

    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # one hot encode target values
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    # reshape data
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize inputs from [0; 255] to [0; 1]
    x_train = x_train / 255
    x_test = x_test / 255

    print('x_train.shape=', x_train.shape)
    print('x_test.shape=', x_test.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8)

    return x_train, y_train, x_test, y_test, x_val, y_val


# draw the training statistics
def plot_training_metrics(choosen_model, history):
    # directory to save the plots
    path = os.path.join('results/mnist', str(choosen_model))
    if not os.path.exists(path):
        os.mkdir(path)
    # plotting the metrics
    fig = plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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


# visualize  wrong predictions
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


# save model

# save model
def save_model(model, model_name):
    # saving the model
    save_dir = "results/mnist/"+str(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = str(model_name) + '.h5'
    model_path = os.path.join(save_dir, file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


# first baseline model
def Model1():
    """
     First baseline model ( simpler one)
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    optm = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# define cnn model
def Model2():
    """
    Second model with more convolutional layers and
    without any regularaization
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# third model
def Model3():
    """
     This is quite interesting
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Selects specific model
def select_model(choice):
    """
    Creates model instance according to
    choice
    """
    model = None
    if choice == 1:
        model = Model1()
    elif choice == 2:
        model = Model2()
    elif choice == 3:
        model = Model3()

    return model


# Function to Run the model
def run_program():
    # load the dataset
    x_train, y_train, x_test, y_test, x_val, y_val = load_and_process_data()

    # define model
    choice = 3  # 1, 2, or 3
    model = select_model(choice)
    model.summary()

    # train model
    # Hyperparameters
    batch_size = 128
    epochs = 1
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val))

    # display training details
    plot_training_metrics(choice, hist)

    # evaluate model
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss: ", loss)
    print("Test Accuracy", acc)

    # save model
    save_model(model, choice)


if __name__ == "__main__":
    # ran everthing from here
    run_program()

    # show plots
    plt.show()
