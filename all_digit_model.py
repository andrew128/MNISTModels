import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
import time
import numpy as np
'''
Model architecture:
- Single convolution layer with max pooling and a single hidden layer
- uses adam optimizer and sparse categorical cross entropy asloss function

Code soure:
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
'''
def get_model():
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

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    return (x_train, y_train, x_test, y_test)
    
def main():
    # (60000, 28, 28, 1) (60000,) (10000, 28, 28, 1) (10000,)
    x_train, y_train, x_test, y_test = get_data()

    accuracies = []
    train_times = []
    test_times = []

    confusion_matrices = None

    num_epochs = 20
    for i in range(num_epochs):
        model = get_model()
        before_train = time.time()
        model.fit(x=x_train,y=y_train, epochs=1)
        before_test = time.time()
        accuracy = model.evaluate(x_test, y_test)
        after_test = time.time()

        # Calculate the confusion matrix
        predictions = model.predict(x_test).argmax(axis=1)
        cm = confusion_matrix(y_test, predictions)

        if i == 0:
            confusion_matrices = cm
        else:
            confusion_matrices = np.add(confusion_matrices, cm)

        accuracies.append(accuracy)
        train_times.append(before_test - before_train)
        test_times.append(after_test - before_test)

    print(confusion_matrices / num_epochs)
    print(accuracies, train_times, test_times)

if __name__ == '__main__':
    main()