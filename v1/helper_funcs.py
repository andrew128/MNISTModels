import tensorflow as tf
import numpy as np

def get_data():
    '''
    Get the MNIST data separated into training and testing inputs and labels.
    '''
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
    This function gets the average of the x highest predicted probabilities for when
    the prediction is correct or not depending on the input correct param.
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

def get_prob_distr_cases(prediction_probs, labels, actual, expected):
    '''
    This function gets the probability distributions for when the predicted label
    is the actual and the label is the expected.
    :param prediction_probs:
    :param labels:
    :param actual:
    :param expected:
    :return: list of probability distributions
    '''
    output = []

    predictions = prediction_probs.argmax(axis=1)

    for i in range(predictions.shape[0]):
        if labels[i] == expected and predictions[i] == actual:
            output.append(prediction_probs[i].tolist())

    return output

def get_prob_distr_based_on_correct(prediction_probs, labels, correct=True):
    '''
    Get the probability distributions and returns the distributions that are either
    correct or incorrect predictions depending on the input param.
    '''
    output = []

    predictions = prediction_probs.argmax(axis=1)

    for i in range(predictions.shape[0]):
        if correct and predictions[i] == labels[i]:
            output.append(prediction_probs[i].tolist())
        if not correct and predictions[i] != labels[i]:
            output.append(prediction_probs[i].tolist())

    return output
