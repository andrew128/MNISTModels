import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.models as models

def train_and_save_models(x_train, y_train):
    # Train each of the models 
    input_shape = (28, 28, 1)

    # No hidden layers (printistic regression)
    l0_model = models.get_trained_l0_all_digit_model(x_train, y_train)
    l0_model.save('models/l0_model')

    # 1 hidden layer
    l1_model = models.get_trained_l1_all_digit_model(x_train, y_train)
    l1_model.save('models/l1_model')

    # 1 conv layer, 1 hidden layer
    l2_model = models.get_trained_l2_all_digit_model(x_train, y_train, input_shape)
    l2_model.save('models/l2_model')

    # 2 conv layers, 1 hidden layer
    l3_model = models.get_trained_l3_all_digit_model(x_train, y_train, input_shape)
    l3_model.save('models/l3_model')

    # 3 conv layers, 1 hidden layer
    l4_model = models.get_trained_l4_all_digit_model(x_train, y_train, input_shape)
    l4_model.save('models/l4_model')

def run_combinations(simple_model, complex_model, x_data, y_data):
    '''
    Attempt all confidence values in 0:0.1:1
    Store accuracy and time for each confidence value
    '''
    conf_values = np.arange(0, 1.0, 0.1)
    conf_values = np.append(conf_values, np.arange(0.91, 1.01, 0.01));

    accuracies = []
    times = []
    simplePercents = []
    for conf_value in conf_values:
        accuracy, time, simplePercent = run_combined(simple_model, complex_model, x_data, y_data, conf_value)
        accuracies.append(accuracy)
        times.append(time)
        simplePercents.append(simplePercent)
    
    return accuracies, times, simplePercents

def run_combined(simple_model, complex_model, inputs, labels, conf_value):
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
    complex_indices = np.where(simple_highest_probs < conf_value, indices, None)
    complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
    complex_indices = np.asarray(complex_indices, dtype=np.int64)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(complex_model.predict(complex_inputs), axis=1)
    # -----------------------------------
    # Select simple
    simple_indices = np.where(simple_highest_probs >= conf_value, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    reduced_simple_probs = np.take(simple_probs, simple_indices, axis=0)
    simple_preds = reduced_simple_probs.argmax(axis=1)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, simple_indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)

    simplePercent = simple_indices.shape[0] / (simple_indices.shape[0] + complex_indices.shape[0])

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simplePercent

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()

    #train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    l1_model = tf.keras.models.load_model('models/l1_model')
    l2_model = tf.keras.models.load_model('models/l2_model')
    l3_model = tf.keras.models.load_model('models/l3_model')
    # l4_model = tf.keras.models.load_model('models/l4_model')

    l1_time = 0
    l2_time = 0
    l3_time = 0

    for i in range(5):
        before_time = time.time()
        accuracy = l1_model.predict(x_test)
        l1_time += time.time() - before_time
        print(l1_time)

        before_time = time.time()
        accuracy = l2_model.predict(x_test)
        l2_time += time.time() - before_time

        before_time = time.time()
        accuracy = l3_model.predict(x_test)
        l3_time += time.time() - before_time

    print("L1 Time:", l1_time / 5)
    print("L2 Time:", l2_time / 5)
    print("L3 Time:", l3_time / 5)

    l1_l2_accuracies = []
    l1_l3_accuracies = []
    l2_l3_accuracies = []
    l1_l2_times = []
    l1_l3_times = []
    l2_l3_times = []
    l1_l2_percents = []
    l1_l3_percents = []
    l2_l3_percents = []

    for i in range(5):
        print("Run l1 l2... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l1_model, l2_model, x_test, y_test)
        l1_l2_accuracies.append(accuracies)
        l1_l2_times.append(times)
        l1_l2_percents.append(simplePercents)

        print("Run l1 l3... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l1_model, l3_model, x_test, y_test)
        l1_l3_accuracies.append(accuracies)
        l1_l3_times.append(times)
        l1_l3_percents.append(simplePercents)

        print("Run l2 l3... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l2_model, l3_model, x_test, y_test)
        l2_l3_accuracies.append(accuracies)
        l2_l3_times.append(times)
        l2_l3_percents.append(simplePercents)

    print("L1 L2 Accuracies:", np.mean(l1_l2_accuracies, axis=0))
    print("L1 L2 Times:", np.mean(l1_l2_times, axis=0))
    print("L1 L2 Simple Percents", np.mean(l1_l2_percents, axis=0))
    print("L1 L3 Accuracies:", np.mean(l1_l3_accuracies, axis=0))
    print("L1 L3 Times:", np.mean(l1_l3_times, axis=0))
    print("L1 L3 Simple Percents", np.mean(l1_l3_percents, axis=0))
    print("L2 L3 Accuracies:", np.mean(l2_l3_accuracies, axis=0))
    print("L2 L3 Times:", np.mean(l2_l3_times, axis=0))
    print("L2 L3 Simple Percents", np.mean(l2_l3_percents, axis=0))

    # print("Run l1 l2...")
    # accuracies, times = run_combinations(l1_model, l2_model, x_test, y_test)
    # print(accuracies)
    # print(times)

    # print("Run l1 l3...")
    # accuracies, times = run_combinations(l1_model, l3_model, x_test, y_test)
    # print(accuracies)
    # print(times)

    # print("Run l2 l3...")
    # accuracies, times = run_combinations(l2_model, l3_model, x_test, y_test)
    # print(accuracies)
    # print(times)

if __name__ == '__main__':
    main()