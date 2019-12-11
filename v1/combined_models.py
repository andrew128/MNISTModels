import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle

import helper_funcs as helpers
from models import *

def get_combined_model_accuracy(confidence_bound=0.56):
    '''
    Get the accuracy using both the simple and the combined.
    When the simple's highest prediction value is less than the confidence_bound input,
    the complex model's prediction is used.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    complex_accuracies = []
    simple_accuracies = []
    combined_accuracies = []
    for i in range(10):
        # trained_complex_all_digit_model = get_trained_complex_all_digit_model(x_train, y_train)
        # trained_complex_all_digit_model.save('trained_complex_all_digit_model_' + str(i))

        trained_complex_all_digit_model = tf.keras.models.load_model('trained_complex_all_digit_model_' + str(i))

        # trained_simple_all_digit_model = get_trained_simple_all_digit_model(x_train, y_train)
        # trained_simple_all_digit_model.save('trained_simple_all_digit_model_' + str(i))

        trained_simple_all_digit_model = tf.keras.models.load_model('trained_simple_all_digit_model_' + str(i))

        complex_accuracy = trained_complex_all_digit_model.evaluate(x_test, y_test)[1]
        # print('Complex Accuracy:', complex_accuracy)
        complex_accuracies.append(complex_accuracy)

        simple_accuracy = trained_simple_all_digit_model.evaluate(x_test, y_test)[1]
        # print('Simple Accuracy:', simple_accuracy)
        simple_accuracies.append(simple_accuracy)

        complex_probs = trained_complex_all_digit_model.predict(x_test)
        complex_highest_probs = np.amax(complex_probs, axis=1)
        complex_preds = complex_probs.argmax(axis=1)

        simple_probs = trained_simple_all_digit_model.predict(x_test)
        simple_highest_probs = np.amax(simple_probs, axis=1)
        simple_preds = simple_probs.argmax(axis=1)

        combined_predictions = np.where(simple_highest_probs > confidence_bound, simple_preds, complex_preds)
        correct_predictions = tf.equal(combined_predictions, y_test)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        combined_accuracies.append(accuracy.numpy())
        print('Combined Accuracy:', accuracy.numpy())
    
    data = {}
    data['complex_' + str(confidence_bound)] = []
    data['complex_' + str(confidence_bound)].append(np.median(complex_accuracies))
    data['complex_' + str(confidence_bound)].append(np.average(complex_accuracies))

    data['simple_' + str(confidence_bound)] = []
    data['simple_' + str(confidence_bound)].append(np.median(simple_accuracies))
    data['simple_' + str(confidence_bound)].append(np.average(simple_accuracies))

    data['combined_' + str(confidence_bound)] = []
    data['combined_' + str(confidence_bound)].append(np.median(combined_accuracies))
    data['combined_' + str(confidence_bound)].append(np.average(combined_accuracies))
    # data['complex_metadata'] = helpers.get_metadata(complex_accuracies)

    # print('Complex Accuracies:', complex_accuracies)
    # print(helpers.get_metadata(complex_accuracies))

    # print('Simple Accuracies:', simple_accuracies)
    # print(helpers.get_metadata(simple_accuracies))

    # print('Combined Accuracies:', combined_accuracies)
    # print(helpers.get_metadata(combined_accuracies))
    
    return data
    

def main():
    accuracies = []
    # confidence_vals = [0.4, 0.56, 0.6, 0.8]
    confidence_vals = np.arange(0, 1, 0.1)
    for confidence_val in confidence_vals:
        # print('----------------------------')
        # print('Confidence Val:', confidence_val)
        # get_combined_model_accuracy
        # print()

        accuracies.append(get_combined_model_accuracy(confidence_val))

    print(accuracies)

if __name__ == '__main__':
    main()