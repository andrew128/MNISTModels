import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

import helper_funcs as helpers
from models import *

'''
This file gets the histograms on the highest predicted probabilities.
'''

def get_highest_percentages(model, inputs, labels, correct=True):
    '''
    Returns the highest percentages for when predicted correctly or incorrectly
    depending on the correct boolean.
    '''
    prediction_probs = model.predict(inputs)
    filtered_distributions = np.asarray(helpers.get_prob_distr_based_on_correct(prediction_probs, labels, correct))
    highest_percentages = np.amax(filtered_distributions, axis=1)
    return highest_percentages

def gen_histogram(percentage_data):
    bins = np.arange(1.0, step=0.03)
    # _ = plt.hist(percentage_data, bins='auto')
    _ = plt.hist(percentage_data, bins=bins)
    plt.show()

def get_metadata(percentage_data):
    '''
    Get data about percentage data.
    '''
    print('Min', np.amin(percentage_data))
    print('Max', np.amax(percentage_data))
    print('Median', np.median(percentage_data))
    print('Mean', np.mean(percentage_data))
    print('Std. Deviation', np.std(percentage_data))
    print('Variance', np.var(percentage_data))

def main():
    x_train, y_train, x_test, y_test = helpers.get_data()

    # Complex model with CNN
    # trained_complex_all_digit_model = get_trained_complex_all_digit_model(x_train, y_train)
    # trained_complex_all_digit_model.save('trained_complex_all_digit_model')
    trained_complex_all_digit_model = tf.keras.models.load_model('trained_complex_all_digit_model')
    # accuracy = trained_complex_all_digit_model.evaluate(x_test, y_test)
    # print('model accuracy', accuracy)

    # Single Dense Layer
    # trained_simple_all_digit_model = get_trained_simple_all_digit_model(x_train, y_train)
    # trained_simple_all_digit_model.save('trained_simple_all_digit_model')
    trained_simple_all_digit_model = tf.keras.models.load_model('trained_simple_all_digit_model')
    # print('Accuracy', trained_simple_all_digit_model.evaluate(x_test, y_test))

    highest_percentages_correct = get_highest_percentages(trained_simple_all_digit_model, x_test, y_test, correct=True)
    # highest_percentages_correct = get_highest_percentages(trained_complex_all_digit_model, x_test, y_test, correct=False)
    # unique, counts = np.unique(highest_percentages_correct, return_counts=True)
    # print(unique)
    get_metadata(highest_percentages_correct)
    gen_histogram(highest_percentages_correct)

if __name__ == '__main__':
    main()