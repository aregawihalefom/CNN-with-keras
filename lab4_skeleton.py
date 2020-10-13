from __future__ import print_function

# The two folloing lines allow to reduce tensorflow verbosity
import os

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '1'  # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING
# msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np

print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test.shape=', y_test.shape)

# To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
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

num_classes = 10

# Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1


# Let start our work: creating a convolutional neural network

##### TO COMPLETE

# Define Network

def Model():
    # creating model instance
    model = Sequential()

    # first Convolutuional layers
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

    # Maxpooling Layers
    model.add(MaxPooling2D(2, 2))

    # Flatten Layers
    model.add(Flatten())

    # Linear Layer
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

    # Final Layer with softmax activation
    model.add(Dense(10, activation='softmax'))

    # complile model
    optm = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Model2():

    model2 = Sequential()
    model2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',
                      input_shape=(28, 28, 1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dense(num_classes, activation='softmax'))

    optm = Adam(learning_rate=0.02)
    model2.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    return model2


def Model3(optimizer='Adam', lr=0.001, momentum=0.9):

    model3 = Sequential()
    model3.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',
                      input_shape=(28, 28, 1)))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Flatten())

    model3.add(Dense(100, activation='relu'))
    model3.add(Dense(num_classes, activation='softmax'))

    optm = Adam(learning_rate=lr)
    model3.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

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


choose_model = 2
# crate the model
if choose_model == 1:
    model = Model()
    model_name = 'mnist-cnn-1'
elif choose_model == 2:
    model = Model2()
    model_name = 'mnist-cnn-2'
elif choose_model == 3:
    model = Model3('Adam', lr=0.1)
    model_name = 'mnist-cnn-2'


# Hyper parameters
batch_size = 128
epochs = 10
validation_split = 0.25


#Do data Augmentation
generator=dataAugmentation(x_train,y_train,True)

# train model now
history = train_model(model, x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

# visualize training statistics
plot_training_metrics(choose_model,history)

# evaluate model
evaluate_model(model, X_test=x_test, Y_test=y_test)

# visualize correct and wrong predictions
visualize_correct_and_wrong_predictions(choose_model, model, X_test=x_test, y_test=y_test)

# save final model
save_model(model, model_name)
plt.show()
