import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import time
import numpy as np
# from sklearn.metrics import confusion_matrix
import helper_funcs as helpers

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
    
def main():
    # (60000, 28, 28, 1) (60000,) (10000, 28, 28, 1) (10000,)
    x_train, y_train, x_test, y_test = helpers.get_data()

    # -------------------------------
    # Train model to recognize digit 8
    # curr_y_train = np.where(y_train == 8, 1, 0)

    # model = get_model()
    # model.fit(x=x_train,y=curr_y_train, epochs=1)

    # predictions = model.predict(x_test)
    # print(predictions)
    # predictions = predictions.T[1]
    # predictions = np.expand_dims(predictions, axis=1)
    # print(predictions)
    # -------------------------------

    # accuracies = []
    # train_times = []
    # test_times = []

    # average_highest_probs_correct = []
    # average_highest_probs_incorrect = []

    # num_classes = 10
    # confusion_matrices = [None] * num_classes

    # num_epochs = 5
    # for i in range(num_epochs):
    #     print('Epoch', i)
    #     all_predicted = None
    #     models = []

    #     before_train = time.time()

    #     # Train the models.
    #     for j in range(10):
    #         # Shape data to be true for only the current i
    #         curr_y_train = np.where(y_train == j, 1, 0)

    #         model = get_model()
    #         models.append(model)
    #         model.fit(x=x_train,y=curr_y_train, epochs=1)

    #         # Calculate the confusion matrix
    #         # predictions = model.predict(x_test).argmax(axis=1)
    #         # cm = confusion_matrix(np.where(y_test == j, 1, 0), predictions)

    #         # if i == 0:
    #         #     confusion_matrices[j] = cm
    #         # else:
    #         #     confusion_matrices[j] = np.add(confusion_matrices[j], cm)

    #     before_test = time.time()

    #     for j, model in enumerate(models):
    #         # Make predictions for x_test and get the probabilities
    #         # predicted that each input is i
    #         predictions = model.predict(x_test)
    #         # Get prediction of current digit
    #         predictions = predictions.T[1]
    #         predictions = np.expand_dims(predictions, axis=1)
    #         if j == 0:
    #             all_predicted = predictions
    #         else:
    #             all_predicted = np.append(all_predicted, predictions, axis=1)

    #     # Get the indices corresponding to the highest predictions for each class.
    #     all_predicted_labels = np.argmax(all_predicted, axis=1)

    #     average_highest_probs_correct.append(helpers.get_average_highest(all_predicted, all_predicted_labels, y_test, 3, True))
    #     average_highest_probs_incorrect.append(helpers.get_average_highest(all_predicted, all_predicted_labels, y_test, 3, False))

    #     num_correct = np.sum(all_predicted_labels == y_test)
    #     final_accuracy = num_correct / float(y_test.shape[0])
    #     # after_test = time.time()

    #     accuracies.append(final_accuracy)
        # train_times.append(before_test - before_train)
        # test_times.append(after_test - before_test)
    
    # for cm in confusion_matrices:
    #     cm = cm / num_epochs
    
    # print(confusion_matrices)
    # print(accuracies, train_times, test_times)
    # print(accuracies)
    # print(average_highest_probs_correct)
    # print(average_highest_probs_incorrect)

if __name__ == '__main__':
    main()