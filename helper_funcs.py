import tensorflow as tf
import numpy as np

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

def get_prob_distr(prob_data, predictions, labels, num_preds, correct=True):
    '''
    :param prob_data:
    :param predictions:
    :param labels:
    :param num_preds:
    :param correct:
    :return:
    '''
    output_prob_distr = []

    for i in range(predictions.shape[0]):
        if len(output_prob_distr) >= num_preds:
            break
        
        if predictions[i] == labels[i] and correct:
            output_prob_distr.append(prob_data[i])
        elif predictions[i] != labels[i] and not correct:
            output_prob_distr.append(prob_data[i])

    return output_prob_distr

def get_average_highest(prob_data, predictions, labels, x_highest, correct=True):
    '''
    :param prob_data:
    :param predictions:
    :param labels:
    :param x_highest:
    :param correct:
    :return:
    '''
    average_highest_probs = np.zeros(x_highest)
    num_preds = 0

    for i in range(predictions.shape[0]):
        sorted_prbs = np.sort(prob_data[i])[::-1]
        
        if predictions[i] == labels[i] and correct:
            average_highest_probs = np.add(average_highest_probs, sorted_prbs[:x_highest])
            num_preds += 1
        elif predictions[i] != labels[i] and not correct:
            average_highest_probs = np.add(average_highest_probs, sorted_prbs[:x_highest])
            num_preds += 1

    return average_highest_probs / num_preds
