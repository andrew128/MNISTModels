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
    p_yy, p_yn, p_ny, p_nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top  = run_combined(simple_model, complex_model, x_data, y_data)
    return p_yy, p_yn, p_ny, p_nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top

def run_combined(simple_model, complex_model, inputs, labels):
    '''
    Runs both models on the input data with the input confidence value
    '''
    before = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_preds = simple_probs.argmax(axis=1)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    complex_probs = complex_model.predict(inputs)
    complex_preds = complex_probs.argmax(axis=1)

    simple_results = tf.equal(simple_preds, labels).numpy()
    complex_results = tf.equal(complex_preds, labels).numpy()

    yy = 0
    yn = 0
    ny = 0
    nn = 0
    yy_values = []
    yn_values = []
    ny_values = []
    nn_values = []
    l1_values = []
    ny_top = []
    for i in range(simple_results.shape[0]):
        second_highest = np.partition(simple_probs[i], -2)[-2]
        if simple_results[i] and complex_results[i]:
            yy += 1
            yy_values.append(simple_highest_probs[i] - second_highest)
            l1_values.append(simple_highest_probs[i] - second_highest)
        elif simple_results[i] and (not complex_results[i]):
            yn += 1
            yn_values.append(simple_highest_probs[i] - second_highest)
            l1_values.append(simple_highest_probs[i] - second_highest)
        elif (not simple_results[i]) and complex_results[i]:
            ny += 1
            ny_values.append(simple_highest_probs[i] - second_highest)
            ny_top.append(simple_highest_probs[i])
        else:
            nn += 1
            nn_values.append(simple_highest_probs[i] - second_highest)

    y_sum = yy + yn + ny + nn
    # return yy / y_sum, yn / y_sum, ny / y_sum, nn / y_sum
    return yy, yn, ny, nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top

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

    p_yy, p_yn, p_ny, p_nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top = run_combinations(l1_model, l2_model, x_test, y_test)
    print("L1 L2 values:", p_yy, p_ny, p_yn, p_nn)
    #print("YY:", yy_values)
    print("YN:", yn_values)
    print("NY:", ny_values)
    print("NN:", nn_values)
    #print("L1: ", l1_values[:1000])

    p_yy, p_yn, p_ny, p_nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top = run_combinations(l1_model, l3_model, x_test, y_test)
    print("L1 L3 values:", p_yy, p_ny, p_yn, p_nn)
    #print("YY:", yy_values)
    print("YN:", yn_values)
    print("NY:", ny_values)
    print("NN:", nn_values)
    print("Top:", ny_top)
    #print("L1: ", l1_values[:1000])

    p_yy, p_yn, p_ny, p_nn, yy_values, yn_values, ny_values, nn_values, l1_values, ny_top = run_combinations(l2_model, l3_model, x_test, y_test)
    print("L2 L3 values:", p_yy, p_ny, p_yn, p_nn)
    #print("YY:", yy_values)
    print("YN:", yn_values)
    print("NY:", ny_values)
    print("NN:", nn_values)
    #print("L1: ", l1_values[:1000])


if __name__ == '__main__':
    main()