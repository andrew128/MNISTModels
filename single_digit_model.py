import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import time
'''
Model architecture:
'''
def get_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(2,activation=tf.nn.softmax))
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

    for i in range(20):
        all_predicted = None
        models = []

        before_train = time.time()
        for i in range(10):
            # Shape data to be true for only the current i
            curr_y_train = np.where(y_train == i, 1, 0)

            model = get_model()
            models.append(model)
            model.fit(x=x_train,y=curr_y_train, epochs=1)

        before_test = time.time()

        for i, model in enumerate(models):
            # Make predictions for x_test and get the probabilities
            # predicted that each input is i
            predictions = model.predict(x_test)
            # Get prediction of current digit
            predictions = predictions.T[1]
            predictions = np.expand_dims(predictions, axis=1)
            if i == 0:
                all_predicted = predictions
            else:
                all_predicted = np.append(all_predicted, predictions, axis=1)

        # Get the indices corresponding to the highest predictions for each class.
        all_predicted_labels = np.argmax(all_predicted, axis=1)
        num_correct = np.sum(all_predicted_labels == y_test)
        final_accuracy = num_correct / y_test.shape[0]
        after_test = time.time()

        accuracies.append(final_accuracy)
        train_times.append(before_test - before_train)
        test_times.append(after_test - before_test)
    
    print(accuracies, train_times, test_times)

if __name__ == '__main__':
    main()