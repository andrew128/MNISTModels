import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
import time
import multiprocessing

import helper_funcs as helpers
from models import *

def test_combined():
    '''
    Computes simple probabilties for all inputs ahead of time.
    Selects indices corresponding to simple prob distributions where highest
    value is below a certain confidence value and makes complex predictions
    on those inputs.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    confidence_values = np.arange(0.1, 1, 0.1)

    times = []
    accuracies = []

    num_epochs = 1

    for confidence_value in confidence_values:
        confidence_value = 0.1
        print('Confidence value:', confidence_value)
        curr_conf_val_total_time = 0
        curr_conf_val_total_accuracy = 0

        for i in range(num_epochs):
            print('Iteration:', i)
            trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model_' + str(i))
            trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model_' + str(i))

            before_time = time.time()
            # -----------------------------------
            # All Simple
            simple_probs = trained_simple_all_digit_model.predict(x_test)
            simple_highest_probs = np.amax(simple_probs, axis=1)

            # -----------------------------------
            # Complex predictions: get inputs of complex predictions
            indices = [i for i in range(y_test.shape[0])]
            complex_indices = np.where(simple_highest_probs < confidence_value, indices, None)
            complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
            complex_indices = np.asarray(complex_indices, dtype=np.int64)

            complex_inputs = np.take(x_test, complex_indices, axis=0)
            complex_preds = np.argmax(trained_complex_all_digit_model.predict(complex_inputs), axis=1)

            # -----------------------------------
            # Select simple
            simple_indices = np.where(simple_highest_probs >= confidence_value, indices, None)
            simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
            simple_indices = np.asarray(simple_indices, dtype=np.int64)

            reduced_simple_probs = np.take(simple_probs, simple_indices, axis=0)
            simple_preds = reduced_simple_probs.argmax(axis=1)

            # -----------------------------------
            # Labels
            complex_labels = np.take(y_test, complex_indices, axis=0)
            simple_labels = np.take(y_test, simple_indices, axis=0)

            # -----------------------------------
            correct_predictions = tf.concat([tf.equal(complex_preds, complex_labels), tf.equal(simple_preds, simple_labels)], axis=0)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            duration = time.time() - before_time
            # -----------------------------------
            curr_conf_val_total_accuracy += accuracy.numpy()
            curr_conf_val_total_time += duration

        accuracies.append(curr_conf_val_total_accuracy / num_epochs)
        times.append(curr_conf_val_total_time / num_epochs)
        print(accuracies, times)
        
    print('--------------------')
    print('Accuracies:', accuracies)
    print('Times:', times)
    

def get_combined_model_accuracy_slow(confidence_bound=0.56):
    '''
    Uses a for loop to get predictions.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    combined_accuracies = []
    combined_times = []
    for i in range(1):
        print('Iteration:', i)
        trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model_' + str(i))

        trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model_' + str(i))

        before_time = time.time()

        # Combined
        simple_probs = trained_simple_all_digit_model.predict(x_test)
        simple_highest_probs = np.amax(simple_probs, axis=1)

        num = 0
        for i in range(simple_highest_probs.shape[0]):
            if simple_highest_probs[i] < confidence_bound:
                num += 1
        print('slow num', num)

        simple_preds = simple_probs.argmax(axis=1)

        combined_predictions = []
        simple_num = 0
        complex_num = 0
        for i in range(y_test.shape[0]):
            label = y_test[i]
            if simple_highest_probs[i] > confidence_bound:
                combined_predictions.append(simple_preds[i])
                simple_num += 1
            else:
                complex_pred = trained_complex_all_digit_model.predict(np.expand_dims(x_test[i], axis=0)).argmax(axis=1)[0]
                combined_predictions.append(complex_pred)
                complex_num += 1

        correct_predictions = tf.equal(combined_predictions, y_test)

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        after_time = time.time()

        combined_accuracies.append(accuracy.numpy())

        combined_times.append(after_time - before_time)

        print('accuracy:', accuracy.numpy())
        print('time:', after_time - before_time)
    
    print('=========================')
    print('Combined (Slow):')
    print('Average Accuracy:', np.average(combined_accuracies))
    print('Average Time:', np.average(combined_times))
    print('=========================')
    

def get_combined_model_accuracy(confidence_bound=0.56):
    '''
    Get the accuracy using both the simple and the combined.
    When the simple's highest prediction value is less than the confidence_bound input,
    the complex model's prediction is used.
    Calculates both the simple and the complex predictions for all inputs.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    combined_accuracies = []
    combined_times = []

    for i in range(1):
        trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model_' + str(i))

        trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model_' + str(i))

        before_time = time.time()

        complex_probs = trained_complex_all_digit_model.predict(x_test)
        complex_highest_probs = np.amax(complex_probs, axis=1)
        complex_preds = complex_probs.argmax(axis=1)

        simple_probs = trained_simple_all_digit_model.predict(x_test)
        simple_highest_probs = np.amax(simple_probs, axis=1)
        simple_preds = simple_probs.argmax(axis=1)

        combined_predictions = np.where(simple_highest_probs > confidence_bound, simple_preds, complex_preds)
        correct_predictions = tf.equal(combined_predictions, y_test)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        after_time = time.time()

        combined_accuracies.append(accuracy.numpy())
        combined_times.append(after_time - before_time)

        # print('Combined Accuracy:', accuracy.numpy())
        
    
    print('===========================')
    print('Combined (fast)')
    print('Average Accuracy:', np.average(combined_accuracies))
    print('Average Time:', np.average(combined_times))
    print('===========================')

    
def get_combined_model_data_splits():
    '''
    Prints arrays where each index contains the times and accuracies of the combined model for a given data split.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()
    trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
    trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')

    combined_times = []
    combined_accuracies = []

    data_splits = np.arange(0.1, 1, 0.1) # [0.1, 0.9]

    # For each test split, run simple, complex, and combined predictions.
    for data_split in data_splits:

        total_combined_time = 0
        total_combined_accuracy = 0

        num_epochs = 10
        for i in range(num_epochs):
            # Shuffle data
            indices = tf.random.shuffle(tf.range(y_test.shape[0]))
            x_test = tf.gather(x_test, indices)
            y_test = tf.gather(y_test, indices)

            # Split data
            split_index = tf.cast(data_split * y_test.shape[0], tf.int32)

            simple_x = x_test[:split_index]
            simple_y = y_test[:split_index]

            complex_x = x_test[split_index:]
            complex_y = y_test[split_index:]

            # Combined
            before_time = time.time()
            # Complex portion
            complex_probs = trained_complex_all_digit_model.predict(complex_x)
            complex_preds = complex_probs.argmax(axis=1)
            complex_output = tf.equal(complex_preds, complex_y)
            # Simple portion
            simple_probs = trained_simple_all_digit_model.predict(simple_x)
            simple_preds = simple_probs.argmax(axis=1)
            simple_output = tf.equal(simple_preds, simple_y)
            # Combined
            combined_output = tf.concat([complex_output, simple_output], axis=0)
            combined_accuracy = tf.reduce_mean(tf.cast(combined_output, tf.float32))

            after_time = time.time()
            total_combined_time += after_time - before_time
            total_combined_accuracy += combined_accuracy

        combined_times.append(total_combined_time / num_epochs)
        combined_accuracies.append(total_combined_accuracy.numpy() / num_epochs)

    print('Combined Times:', combined_times)
    print('Combined Accuracies:', combined_accuracies)

def get_times_and_accuracies(simple=True):
    '''
    Print the average time and accuracies for the given model on the
    test portion of the MNIST dataset.
    '''
    model_type = './models/trained_simple_all_digit_model_'
    if not simple:
        model_type = './models/trained_complex_all_digit_model_'

    x_train, y_train, x_test, y_test = helpers.get_data()

    total_time = 0
    total_accuracy = 0

    num_epochs = 10
    for i in range(num_epochs):

        model = tf.keras.models.load_model(model_type + str(i))

        # Shuffle data
        indices = tf.random.shuffle(tf.range(y_test.shape[0]))
        x_test = tf.gather(x_test, indices)
        y_test = tf.gather(y_test, indices)

        before_time = time.time()
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        after_time = time.time()
        total_time += after_time - before_time
        total_accuracy += accuracy

    average_time = total_time / num_epochs
    average_accuracy = total_accuracy / num_epochs

    print('Average Time:', average_time)
    print('Average Accuracy:', average_accuracy)

def main():
    # --------------------------
    # Combined
    test_combined()
    # --------------------------
    # Simple 
    # print('Simple --------------------------------')
    # get_times_and_accuracies()
    # ---------------------------
    # Complex
    # print('Complex --------------------------------')
    # get_times_and_accuracies(simple=False)

if __name__ == '__main__':
    main()