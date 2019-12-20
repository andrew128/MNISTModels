import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
import time
import multiprocessing

import helper_funcs as helpers
from models import *
import combined_model_tests as combined_test_funcs

def get_random_accuracy_and_latency():
    combined_test_funcs.get_combined_model_data_splits()

def get_combined_accuracy_and_latency():
    '''
    For a range of accuracies, calculate the confidence value from 
    Tab 8's Accuracy vs Confidence Value graph. Then call helper function to get 
    duration, accuracy, and # of complex calls from given confidence value.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    durations = []
    actual_accuracies = []
    simple_percentage_calls = []
    conf_values = []

    accuracies = np.arange(0.91, 0.98, 0.01)
    acc_conf_val_trendline_coef = [-0.23, 0.375, -0.0884, 0.915]

    num_epochs = 10
    for accuracy in accuracies:
        print('Accuracy:', accuracy)
        # Solve for the inverse to get the confidence value
        # given an accuracy. 
        deep_copy_coef = np.copy(acc_conf_val_trendline_coef)
        deep_copy_coef[deep_copy_coef.shape[0] - 1] -= accuracy
        confidence_value = np.roots(deep_copy_coef).real[1] # Get the second solution.
        # print('Conf value:', confidence_value)
        conf_values.append(confidence_value)
        total_duration = 0
        total_accuracy = 0
        total_simple_percentage_calls = 0

        for i in range(num_epochs):
            print('  Epoch:', i)
            trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model_' + str(i))
            trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model_' + str(i))
            # print(accuracy, confidence_value)
            duration, accuracy, num_complex_calls = get_num_complex_calls(x_test, y_test, confidence_value, \
                trained_simple_all_digit_model, trained_complex_all_digit_model)
        
            num_simple_percentage_call = (y_test.shape[0] - num_complex_calls) / y_test.shape[0]

            total_duration += duration
            total_accuracy += accuracy
            total_simple_percentage_calls += num_simple_percentage_call

        print(total_duration / num_epochs, total_accuracy / num_epochs, total_simple_percentage_calls / num_epochs)
        durations.append(total_duration / num_epochs)
        actual_accuracies.append(total_accuracy / num_epochs)
        simple_percentage_calls.append(total_simple_percentage_calls / num_epochs)
    
    print(durations)
    print(actual_accuracies)
    print(simple_percentage_calls)
    print(conf_values)

def get_num_complex_calls(inputs, labels, confidence_bound, trained_simple_all_digit_model,\
    trained_complex_all_digit_model):
    '''
    For a given confidence value, get the # of times the complex model 
    was called.
    '''
    # trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
    # trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')

    before_time = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = trained_simple_all_digit_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    indices = [i for i in range(inputs.shape[0])]
    complex_indices = np.where(simple_highest_probs < confidence_bound, indices, None)
    complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
    complex_indices = np.asarray(complex_indices, dtype=np.int64)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(trained_complex_all_digit_model.predict(complex_inputs), axis=1)
    # -----------------------------------
    # Select simple
    simple_indices = np.where(simple_highest_probs >= confidence_bound, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    reduced_simple_probs = np.take(simple_probs, simple_indices, axis=0)
    simple_preds = reduced_simple_probs.argmax(axis=1)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, simple_indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)

    duration = time.time() - before_time
    accuracy = tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy()
    num_complex_calls = complex_indices.shape[0]

    return (duration, accuracy, num_complex_calls)

def main():
    # get_random_accuracy_and_latency()
    get_combined_accuracy_and_latency()

if __name__ == '__main__':
    main()