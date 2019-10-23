import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import time
'''
Model architecture:
'''
def get_model():
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    # model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
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

    # ---------------------------------

    # y_train = np.where(y_train == 4, 1, 0)

    # model = get_model()
    # model.fit(x=x_train,y=y_train, epochs=1)

    # predictions = model.predict(x_test)
    # for i in range(5):
    #     print(predictions.T[1][i])
    #     print(y_test[i])
    # ----------------------------------
    # y_train = (y_train == 4) # True for all i's, False for all other digits
    # y_test = (y_test == 4)

    # model = get_model()
    # model.fit(x=x_train,y=y_train, epochs=1)
    # predictions = model.predict(x_test)
    # predicted_labels = np.argmax(predictions, axis=1)
    # print(predicted_labels.shape)
    # print(predictions.shape)
    # print(predictions[0])

    all_predicted = None
    models = []
    for i in range(10):
        # Shape data to be true for only the current i
        # x_train = (x_train == i)
        # y_train = (y_train == i)
        curr_y_train = np.where(y_train == i, 1, 0)

        model = get_model()
        models.append(model)
        model.fit(x=x_train,y=curr_y_train, epochs=1)

        # Make predictions for x_test and get the probabilities
        # predicted that each input is i
        predictions = model.predict(x_test)
        print('output predictions shape', predictions.shape)
        print('first 3 predictions', predictions[0:3])
        print('first 3 labels', y_test[0:3])
        # Get prediction of current digit
        predictions = predictions.T[1]
        # print('specific index shape', predictions.shape)
        predictions = np.expand_dims(predictions, axis=1)
        # print('expanded predictions shape', predictions.shape)
        if i == 0:
            all_predicted = predictions
        else:
        #     print('all predicted shape', all_predicted.shape)
            # print('predictions shape', predictions.shape)
            # print('old all predicted')
            all_predicted = np.append(all_predicted, predictions, axis=1)
            # print('new all predicted shape', all_predicted.shape)
            # all_predicted = np.concatenate((all_predicted, np.array(predictions)), axis=1)

    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')

    # for model in models:
    #     print(model.predict(x_test)[0])


    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # print('-------------------------------')
    # Reshape all_predicted into (num_labels, num_classes)
    # all_predicted = all_predicted.reshape(y_test.shape[0], 10)]
    # print('before', all_predicted.shape)
    # all_predicted = all_predicted.T
    # print('after', all_predicted.shape)

    # for i in range(5):
    #     print('all_predicted shape', all_predicted.shape)
    #     print('first row', all_predicted[i])

    # Get the indices corresponding to the highest predictions for each class.
    all_predicted_labels = np.argmax(all_predicted, axis=1)

    # for i in range(5):
    #     print('all_predicted_labels shape', all_predicted_labels.shape)
    #     print('first row', all_predicted[i])
    #     print('actual label', y_test[i])

    # print('labels shape', y_test.shape)
    # print('all predicted labels shape', all_predicted_labels.shape)
    num_correct = np.sum(all_predicted_labels == y_test)
    print('Final Accuracy:', num_correct / y_test.shape[0])


        # print(predictions.shape)
        # print(predictions[0])

    # model = get_model()
    # before_train = time.time()
    # model.fit(x=x_train,y=y_train, epochs=1)
    # before_test = time.time()
    # model.evaluate(x_test, y_test)
    # after_test = time.time()
    # print('Train time:', before_test - before_train)
    # print('Test time:', after_test - before_test)

if __name__ == '__main__':
    main()