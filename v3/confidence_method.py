import numpy as np
import tensorflow as tf
import time
import helper_funcs as helpers
from models import *

def test_combined(simple_model_level, complex_model_level):
    '''
    Gets data for Times and Accuracies vs Confidence value for the combined model.

    Combined method:
    Computes simple probabilties for all inputs ahead of time.
    Selects indices corresponding to simple prob distributions where highest
    value is below a certain confidence value and makes complex predictions
    on those inputs.
    '''
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()

    y_test = tf.squeeze(y_test)

    x_test = x_test[:5000]
    y_test = y_test[:5000]

    confidence_values = np.arange(0, 1.1, 0.1)

    times = []
    accuracies = []
    percentage_simple_calls = []

    num_epochs = 5

    for confidence_value in confidence_values:
        print('Confidence Value:', confidence_value)
        curr_conf_val_total_time = 0
        curr_conf_val_total_accuracy = 0
        curr_percentage_simple_calls = 0

        for i in range(num_epochs):
            # print('Epoch', i, '/', num_epochs)
            trained_complex_all_digit_model = tf.keras.models.load_model('./models/l' + str(complex_model_level) + '_cifar10_model' + str(i))
            trained_simple_all_digit_model = tf.keras.models.load_model('./models/l' + str(simple_model_level) + '_cifar10_model' + str(i))

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
            if complex_inputs.shape[0] == 0:
                complex_preds = []
            else:
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
            # print('percentage simple', simple_labels.shape[0], y_test.shape[0], simple_labels.shape[0] / y_test.shape[0])
            curr_percentage_simple_calls += (simple_labels.shape[0] / y_test.shape[0])

        accuracies.append(curr_conf_val_total_accuracy / num_epochs)
        times.append(curr_conf_val_total_time / num_epochs)
        percentage_simple_calls.append(curr_percentage_simple_calls / num_epochs)
        
    print('Combined --------------------')
    print('Accuracies:', accuracies)
    print('Times:', times)
    print('Percentage Simple', percentage_simple_calls)

def main():
    print(1, 2)
    test_combined(1, 2)
    print(1, 3)
    test_combined(1, 3)
    print(1, 4)
    test_combined(1, 4)
    print(2, 3)
    test_combined(2, 3)
    print(2, 4)
    test_combined(2, 4)
    print(3, 4)
    test_combined(3, 4)

if __name__ == '__main__':
    main()