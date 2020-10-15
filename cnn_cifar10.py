from __future__ import print_function

# The two folloing lines allow to reduce tensorflow verbosity
import os

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '1'  # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING
# msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


# load and prepocess data
def load_and_process_data():
    """
    This model loads the MNIST dataset from the keras api
    @ return : x_train : training features
               y_train : training labels
               x_test  : test features
               y_test : test labels
               x_val : validation features
               y_val : validation labels

    """
    num_classes = 10

    # load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

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


# define label names
def load_label_names():
    """
    Returns the labels in a string fromat for 10 classes of CIFAR-10
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# draw the training statistics
def plot_training_metrics(choice, history):
    """
    This function plots the training metrics statistics

    """

    # plotting the metrics
    fig = plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper right')
    plt.tight_layout()


# visualize  wrong predictions
def visualize_correct_and_wrong_predictions(choosen_model, model, X_test, y_test):

    """
    This function shows wrongly classified sample images
    @params : model , name of model , test data
    @return : none
    """

    #1. predicted classes
    predicted_classes = model.predict_classes(X_test)

    #2. reverse the true level from one hot encoding back  to normal ture lavel
    y_reversed = np.argmax(y_test, axis=1)

    #3. see which we predicted correctly and which not
    correct_indices = np.nonzero(predicted_classes == y_reversed)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_reversed)[0]

    print()
    print(correct_indices.shape[0], " classified correctly")
    print(incorrect_indices.shape[0], " classified incorrectly")

    # label names
    actual_classes = load_label_names()

    # adapt figure size to accomodate 18 subplots
    plt.figure(figsize=(10, 10))

    # get wrong 9 samples
    wrong_nine = incorrect_indices[:9]

    # plot 9 incorrect predictions
    for i, incorrect in enumerate(wrong_nine):

        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(32, 32, 3), cmap='gray', interpolation='none')
        plt.title(
            "True = {} \n Predicted = {}".format(actual_classes[y_reversed[incorrect]],
                                                 actual_classes[predicted_classes[incorrect]]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(pad=1)
    # plt.savefig('{}/wrongly-classified-model-{}.png'.format(path, choosen_model), format='png')
    # plt.clf()


# save model
def save_model(model, model_name):
    """
     Saves model as .h5 file format
     @params : model , model name
     @ return : None
    """
    # saving the model
    save_dir = "results/cifar10/" + str(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = str(model_name) + '.h5'
    model_path = os.path.join(save_dir, file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


# First baseline model
def Model1(lr):
    """
     First baseline model ( simpler one)
     architecture : conv -> max-pooling -> flatten -> dense -> dense -> dense

    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    optm = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Second model
def Model2(lr):
    """
    Second model with more convolutional layers and
    without any regularaization
    architecture : conv -> conv -> max-pooling -> conv -> conv -> max-pooling ->flatten -> dense -> dense
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Third model
def Model3(lr):
    """
     Third model with more convolutional layers and
     with  regularaization
     Architecture : conv -> conv -> max-pooling -> conv -> conv -> max-pooling ->dropout-> flatten -> dense -> dropout->dense

    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Selects specific model
def select_model(choice, lr):
    """
    Creates model instance according to choice
    @params : choice
    @ return : model
    """
    model = None
    if choice == 1:
        model = Model1(lr)
    elif choice == 2:
        model = Model2(lr)
    elif choice == 3:
        model = Model3(lr)

    return model


# Function to Run the model
def run_program():

    ###################################
    #     Program starts Here         #
    ###################################

    # 1. load the dataset
    x_train, y_train, x_test, y_test, x_val, y_val = load_and_process_data()

    # 2. Define model
    #####################################3
    # Hyperparameters
    batch_size = 128
    epochs = 2
    lr = 0.01
    ######################################

    choice = 3  # 1, 2, or 3
    model = select_model(choice, lr)
    model.summary()

    # 3. train model
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    # 4. display training details
    plot_training_metrics(choice, hist)

    # 5. evaluate model
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss: ", loss)
    print("Test Accuracy", acc)

    #6. show wrongly classified
    visualize_correct_and_wrong_predictions(choice, model, x_test, y_test)

    # 7.. save model
    #save_model(model, choice)

    #8. show plots if any
    plt.show()


if __name__ == "__main__":

    # ran everthing from here
    run_program()


