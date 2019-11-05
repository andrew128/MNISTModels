import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
import time
import numpy as np
<<<<<<< HEAD
import helper_funcs as helpers
=======
import helper_funcs as helper

>>>>>>> 300a1117e37a4026064c08061497a30132ac0aed
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
<<<<<<< HEAD
    
def main():
    # (60000, 28, 28, 1) (60000,) (10000, 28, 28, 1) (10000,)
    x_train, y_train, x_test, y_test = helpers.get_data()
=======

def main():
    # (60000, 28, 28, 1) (60000,) (10000, 28, 28, 1) (10000,)
    x_train, y_train, x_test, y_test = helper.get_data()
>>>>>>> 300a1117e37a4026064c08061497a30132ac0aed

    accuracies = []
    train_times = []
    test_times = []

    confusion_matrices = None

    num_epochs = 1
    for i in range(num_epochs):
        model = get_model()
        before_train = time.time()
        model.fit(x=x_train,y=y_train, epochs=1)
        before_test = time.time()
        accuracy = model.evaluate(x_test, y_test)
        after_test = time.time()

        print('Accuracy:', accuracy)

        prediction_probs = model.predict(x_test) 
        predictions = prediction_probs.argmax(axis=1)

        helpers.get_prob_distr(prediction_probs, predictions, y_test, 3, True)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, predictions)

        if i == 0:
            confusion_matrices = cm
        else:
            confusion_matrices = np.add(confusion_matrices, cm)

        accuracies.append(accuracy)
        train_times.append(before_test - before_train)
        test_times.append(after_test - before_test)

    # print(confusion_matrices / num_epochs)
    # print(accuracies, train_times, test_times)

if __name__ == '__main__':
    main()