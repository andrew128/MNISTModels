import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.models as models

def run_combinations(simple_model, complex_model, x_data, y_data):
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
    threshold_counts = []
    for threshold in thresholds:
        all_conf_values.append(threshold)
        print("Threshold", threshold)
        accuracy, time, simple_percent, threshold_count = run_combined(simple_model, complex_model,\
             x_data, y_data, threshold)

        print("accuracy:", accuracy, "time:", time)
        accuracies.append(accuracy)
        times.append(time)
        simple_percents.append(simple_percent)
        threshold_counts.append(threshold_count)
    
    return all_conf_values, accuracies, times, simple_percents, threshold_counts

def run_combined(simple_model, complex_model, inputs, labels, threshold):
    '''
    Runs both models on the input data with the input confidence value
    '''
    before = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    indices = [i for i in range(inputs.shape[0])]
    complex_indices = []
    threshold_count = 0
    for i in range(inputs.shape[0]):
        second_highest = np.partition(simple_probs[i], -2)[-2]
        diff = simple_highest_probs[i] - second_highest
        if diff < threshold:
            complex_indices.append(i)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(complex_model.predict(complex_inputs), axis=1)
    # -----------------------------------
    # Select simple
    simple_preds = np.argmax(simple_probs, axis=1)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)

    simplePercent = 1 - (len(complex_indices) / len(indices))

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simplePercent, threshold_count

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
    all_model_thresholds = []

    for i in range(1):
        print("Run l1 l2... #" + str(i))
        all_conf_values, accuracies, times, simple_percents, thresholds = run_combinations(l2_model, l3_model, x_test, y_test)
        all_model_accuracies.append(accuracies)
        all_model_times.append(times)
        all_model_simple_percents.append(simple_percents)
        all_model_thresholds.append(thresholds)

    print("All Thresholds:", all_conf_values)
    print("L1 L2 Accuracies:", np.mean(all_model_accuracies, axis=0))
    print("L1 L2 Times:", np.mean(all_model_times, axis=0))
    print("L1 L2 Simple Percents:", np.mean(all_model_simple_percents, axis=0))
    print("L1 L2 Thresholds:", np.mean(all_model_thresholds, axis=0))

if __name__ == '__main__':
    main()