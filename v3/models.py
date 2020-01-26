import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from sklearn.svm import SVC
import helper_funcs as helpers

def get_untrained_l4_all_digit_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def get_trained_l4_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l4_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

def get_untrained_l3_all_digit_model(input_shape):
    # input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Conv2D(28, input_shape=input_shape, kernel_size=(5,5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(28, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(1000,))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10,activation=tf.nn.softmax))


    # Compile the model
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_trained_l3_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l3_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

'''
L2
'''
def get_untrained_l2_all_digit_model(input_shape):
    # input_shape = (28, 28, 1)

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

def get_trained_l2_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l2_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

'''
Simple 10 digit model
'''
def get_untrained_l1_all_digit_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_trained_l1_all_digit_model(inputs, labels, epochs=1):
    model = get_untrained_l1_all_digit_model()
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model
