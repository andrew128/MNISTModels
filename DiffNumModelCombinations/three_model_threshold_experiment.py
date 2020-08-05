import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.models as models

def run_combinations(simple_model, complex_model, most_complex_model, x_data, y_data):
    '''
    Attempt all confidence values in 0:0.1:1
    Store accuracy and time for each confidence value
    '''
    thresholds = np.arange(0.2, 1.0, 0.04)
    thresholds = np.append(thresholds, np.arange(0.965, 1.0, 0.005))

    all_conf_values = []
    accuracies = []
    times = []
    simple_percents = []
    complex_percents = []
    for threshold_1 in thresholds:
        for threshold_2 in thresholds:
            current_conf_values = [threshold_1, threshold_2]
            all_conf_values.append(current_conf_values)
            print("Confidence Values", threshold_1, threshold_2)
            accuracy, time, simple_percent, complex_percent = run_combined(simple_model, complex_model, most_complex_model,\
                 x_data, y_data, threshold_1, threshold_2)

            print("accuracy:", accuracy, "time:", time)
            accuracies.append(accuracy)
            times.append(time)
            simple_percents.append(simple_percent)
            complex_percents.append(complex_percent)
    
    return all_conf_values, accuracies, times, simple_percents, complex_percents

def run_combined(simple_model, complex_model, most_complex_model, inputs, labels, conf_value_1, conf_value_2):
    '''
    Runs all three models on the input data with the input confidence value
    '''
    before = time.time()

    indices = [i for i in range(inputs.shape[0])]
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    simple_indices = np.where(simple_highest_probs >= conf_value_1, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    print('simple indices', simple_indices.shape)

    reduced_simple_probs = np.take(simple_probs, simple_indices, axis=0)
    simple_preds = reduced_simple_probs.argmax(axis=1)
    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    complex_indices = []
    for i in range(inputs.shape[0]):
        second_highest = np.partition(simple_probs[i], -2)[-2]
        diff = simple_highest_probs[i] - second_highest
        if diff < conf_value_1:
            complex_indices.append(i)

    print('complex indices', len(complex_indices))

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    complex_highest_probs = None
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_probs = complex_model.predict(complex_inputs)
        complex_highest_probs = np.amax(complex_probs, axis=1)
        complex_preds = np.argmax(complex_probs, axis=1)

    # -----------------------------------
    # Get most complex inputs
    most_complex_indices = []
    most_complex_preds = []
    if not (complex_highest_probs is None):
        for i in range(complex_inputs.shape[0]):
            second_highest = np.partition(complex_probs[i], -2)[-2]
            diff = complex_highest_probs[i] - second_highest
            if diff < conf_value_2:
                most_complex_indices.append(complex_indices[i])

        most_complex_inputs = np.take(inputs, most_complex_indices, axis=0)
        if most_complex_inputs.shape[0] != 0:
            most_complex_probs = most_complex_model.predict(most_complex_inputs)
            most_complex_highest_probs = np.amax(most_complex_probs, axis=1)
            most_complex_preds = np.argmax(most_complex_probs, axis=1)

    print('most complex indices', len(most_complex_indices))

    # -----------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, simple_indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)
    np.put(combined_preds, most_complex_indices, most_complex_preds)

    simple_percent = len(simple_indices) / (len(simple_indices) + len(complex_indices) + len(most_complex_indices))
    complex_percent = len(complex_indices) / (len(simple_indices) + len(complex_indices) + len(most_complex_indices))

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simple_percent, complex_percent

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()

    # train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    l1_model = tf.keras.models.load_model('models/l1_model')
    l2_model = tf.keras.models.load_model('models/l2_model')
    l3_model = tf.keras.models.load_model('models/l3_model')
    # l4_model = tf.keras.models.load_model('models/l4_model')

    # before_time = time.time()
    # accuracy = l0_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l1_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l2_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l3_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l4_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    all_model_accuracies = []
    all_model_times = []
    all_model_simple_percents = []
    all_model_complex_percents = []

    for i in range(1):
        print("Run l1 l2 l3... #" + str(i))
        all_conf_values, accuracies, times, simple_percents, complex_percents = run_combinations(l1_model, l2_model, l3_model, x_test, y_test)
        all_model_accuracies.append(accuracies)
        all_model_times.append(times)
        all_model_simple_percents.append(simple_percents)
        all_model_complex_percents.append(complex_percents)

    print("All Conf Values:", all_conf_values)
    print("L1 L2 L3 Accuracies:", np.mean(all_model_accuracies, axis=0))
    print("L1 L2 L3 Times:", np.mean(all_model_times, axis=0))
    print("L1 L2 L3 Simple Percents:", np.mean(all_model_simple_percents, axis=0))
    print("L1 L2 L3 Complex Percents:", np.mean(all_model_complex_percents, axis=0))

if __name__ == '__main__':
    main()