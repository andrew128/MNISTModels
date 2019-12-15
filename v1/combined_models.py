import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
import time

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
    
def get_combined_model_latency():
    '''
    Get time the combined model takes to predict the testing set for specific dataset splits.
    Returns arrays where each index contains the times and accuracies for a given data split.
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

            print('=====================')
            print('Split Index:', split_index)
            print('Simple:', simple_x.shape[0])
            print('Complex:', complex_x.shape[0])

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
            # print('output shapes', complex_output.shape, simple_output.shape)
            combined_output = tf.concat([complex_output, simple_output], axis=0)
            combined_accuracy = tf.reduce_mean(tf.cast(combined_output, tf.float32))

            after_time = time.time()
            total_combined_time += after_time - before_time
            total_combined_accuracy += combined_accuracy

        combined_times.append(total_combined_time / num_epochs)
        combined_accuracies.append(total_combined_accuracy.numpy() / num_epochs)

    print('Combined Times:', combined_times)
    print('Combined Accuracies:', combined_accuracies)

def get_times_and_accuracies(model):
    '''
    Print the average time and accuracies for the given model on the
    test portion of the MNIST dataset.
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    total_time = 0
    total_accuracy = 0

    num_epochs = 10
    for i in range(num_epochs):
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
    # get_combined_model_latency()
    # --------------------------
    # Simple 
    trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')
    print('Simple')
    get_times_and_accuracies(trained_simple_all_digit_model)
    # ---------------------------
    # Complex
    trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
    print('Complex')
    get_times_and_accuracies(trained_complex_all_digit_model)

if __name__ == '__main__':
    main()