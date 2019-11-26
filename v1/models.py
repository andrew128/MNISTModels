import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.svm import SVC
import helper_funcs as helpers

'''
Model architecture:
- Single convolution layer with max pooling and a single hidden layer
- uses adam optimizer and sparse categorical cross entropy asloss function

Code source:
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
'''
def get_untrained_complex_all_digit_model():
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_trained_complex_all_digit_model(inputs, labels):
    model = get_untrained_complex_all_digit_model()
    model.fit(x=inputs,y=labels, epochs=1)
    return model

'''
Simple 10 digit model
'''
def get_untrained_simple_all_digit_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_trained_simple_all_digit_model(inputs, labels):
    model = get_untrained_simple_all_digit_model()
    model.fit(x=inputs,y=labels, epochs=1)
    return model

'''
10 digit SVM
'''
def get_10_digit_SVM(inputs, labels):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(inputs, labels)
    return svclassifier

'''
Model architecture:
- Single hidden layer
'''
def get_single_digit_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(2,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model