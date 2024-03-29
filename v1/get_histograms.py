import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle

import helper_funcs as helpers
from models import *

'''
This file gets the histograms on the highest predicted probabilities.
'''

def get_highest_percentages(model, inputs, labels, correct=True, is_keras_model=True):
    '''
    Returns the highest percentages for when predicted correctly or incorrectly
    depending on the correct boolean.
    '''
    # print('Get highest percentages')
    # inputs = inputs[:100]
    # labels = labels[:100]
    if not is_keras_model:
        inputs = np.reshape(inputs, (-1, 784))
        prediction_probs = model.predict_proba(inputs)
    else:
        prediction_probs = model.predict(inputs)
    # print(prediction_probs[:10])
    filtered_distributions = np.asarray(helpers.get_prob_distr_based_on_correct(prediction_probs, labels, correct))
    highest_percentages = np.amax(filtered_distributions, axis=1)
    return highest_percentages

def gen_histogram(percentage_data, title):
    bins = np.append(np.arange(1.0, step=0.03), 1)
    _ = plt.hist(percentage_data, bins=bins)
    plt.suptitle(title)
    plt.xlabel('Percentages')
    plt.ylabel('# of predictions')
    plt.grid(True)
    plt.show()

def get_metadata(percentage_data):
    '''
    Get data about percentage data.
    '''
    metadata = {}
    metadata['Min'] = np.amin(percentage_data)
    metadata['Max'] = np.amax(percentage_data)
    metadata['Median'] = np.median(percentage_data)
    metadata['Mean'] = np.mean(percentage_data)
    metadata['StandardDeviation'] = np.std(percentage_data)
    metadata['Variance'] = np.var(percentage_data)
    return metadata

def get_histogram_data():
    x_train, y_train, x_test, y_test = helpers.get_data()

    # Complex model with CNN
    # trained_complex_all_digit_model = get_trained_complex_all_digit_model(x_train, y_train)
    # trained_complex_all_digit_model.save('trained_complex_all_digit_model')
    trained_complex_all_digit_model = tf.keras.models.load_model('trained_complex_all_digit_model')
    # accuracy = trained_complex_all_digit_model.evaluate(x_test, y_test)
    # print('model accuracy', accuracy)

    complex_all_digit_percentages_correct = get_highest_percentages(trained_complex_all_digit_model, \
                                                                    x_test, y_test, correct=True, is_keras_model=True)
    complex_all_digit_percentages_incorrect = get_highest_percentages(trained_complex_all_digit_model, \
                                                                    x_test, y_test, correct=False, is_keras_model=True)
    # ----------------------------------------------
    # Single Dense Layer
    # trained_simple_all_digit_model = get_trained_simple_all_digit_model(x_train, y_train)
    # trained_simple_all_digit_model.save('trained_simple_all_digit_model')
    trained_simple_all_digit_model = tf.keras.models.load_model('trained_simple_all_digit_model')
    # print('Accuracy', trained_simple_all_digit_model.evaluate(x_test, y_test))

    simple_all_digit_percentages_correct = get_highest_percentages(trained_simple_all_digit_model, \
                                                                    x_test, y_test, correct=True, is_keras_model=True)
    simple_all_digit_percentages_incorrect = get_highest_percentages(trained_simple_all_digit_model, \
                                                                    x_test, y_test, correct=False, is_keras_model=True)
    # ----------------------------------------------
    # SVM Model
    # all_digit_SVM = get_10_digit_SVM(x_train, y_train)
    # saved_all_digit_SVM = pickle.dumps(all_digit_SVM)

    # all_digit_SVM.save('all_digit_SVM')
    # all_digit_SVM = tf.keras.models.load_model('all_digit_SVM')
    # --------------------------------------------------
    complex_all_digit_train_time = helpers.get_model_training_times(trained_complex_all_digit_model, x_train, y_train, 10)
    complex_all_digit_test_time = helpers.get_model_training_times(trained_complex_all_digit_model, x_test, y_test, 10)

    # complex_correct_metadata = get_metadata(complex_all_digit_percentages_correct)
    # complex_incorrect_metadata = get_metadata(complex_all_digit_percentages_incorrect)
    # gen_histogram(complex_all_digit_percentages_correct, 'Complex All Digit Correct')
    # gen_histogram(complex_all_digit_percentages_incorrect, 'Complex All Digit Incorrect')

    simple_all_digit_train_time = helpers.get_model_training_times(trained_simple_all_digit_model, x_train, y_train, 10)
    simple_all_digit_test_time = helpers.get_model_training_times(trained_simple_all_digit_model, x_test, y_test, 10)
    
    # simple_correct_metadata = get_metadata(simple_all_digit_percentages_correct)
    # simple_incorrect_metadata = get_metadata(simple_all_digit_percentages_incorrect)
    # gen_histogram(simple_all_digit_percentages_correct, 'Simple All Digit Correct')
    # gen_histogram(simple_all_digit_percentages_incorrect, 'Simple All Digit Incorrect')

    # print('Complex All Digit Train time', complex_all_digit_train_time)
    # print('Complex All Digit Test time', complex_all_digit_test_time)
    # print('Complex Correct MetaData', complex_correct_metadata)
    # print('Complex Incorrect MetaData', complex_incorrect_metadata)

    # print('Simple All Digit Train time', simple_all_digit_train_time)
    # print('Simple All Digit Test time', simple_all_digit_test_time)
    # print('Simple Correct MetaData', simple_correct_metadata)
    # print('Simple Incorrect MetaData', simple_incorrect_metadata)
    # print(len(complex_all_digit_percentages_correct), len(complex_all_digit_percentages_incorrect), len(simple_all_digit_percentages_correct), len(simple_all_digit_percentages_incorrect))

    print(complex_all_digit_train_time, complex_all_digit_test_time)
    print(simple_all_digit_train_time, simple_all_digit_test_time)

def get_combined_histogram_data(simple_data=True):
    '''
    Get histogram data for the model when the simple model is correct/incorrect
    for when the complex model is correct/incorrect.
    The model is simple if simple_data is True. Otherwise the data is from the complex model
    '''
    x_train, y_train, x_test, y_test = helpers.get_data()

    trained_complex_all_digit_model = tf.keras.models.load_model('trained_complex_all_digit_model')
    trained_simple_all_digit_model = tf.keras.models.load_model('trained_simple_all_digit_model')

    # For each data point in x_test, add probability of simple model to list if simple got right and complex got wrong
    # or simple got wrong and complex got right.
    simple_correct_complex_incorrect = []
    simple_incorrect_complex_correct = []

    complex_probs = trained_complex_all_digit_model.predict(x_test)
    complex_preds = complex_probs.argmax(axis=1)
    simple_probs = trained_simple_all_digit_model.predict(x_test)
    simple_preds = simple_probs.argmax(axis=1)

    for i in range(y_test.shape[0]):
        label = y_test[i]
        complex_pred = complex_preds[i]
        simple_pred = simple_preds[i]

        if simple_data:
            prob = np.max(simple_probs[i])
        else:
            prob = np.max(complex_probs[i])

        if simple_pred == label and complex_pred != label:
            simple_correct_complex_incorrect.append(prob)
            # print(simple_prob)
        elif simple_pred != label and complex_pred == label:
            simple_incorrect_complex_correct.append(prob)
            # print(simple_prob)

    if simple_data:
        type_of_data = 'Simple Data'
    else:
        type_of_data = 'Complex Data'

    gen_histogram(simple_correct_complex_incorrect, 'Simple Correct Complex Incorrect - ' + type_of_data)
    gen_histogram(simple_incorrect_complex_correct, 'Simple Incorrect Complex Correct - ' + type_of_data)
    print(get_metadata(simple_correct_complex_incorrect))
    print('--------------------------')
    print(get_metadata(simple_incorrect_complex_correct))
    print('--------------------------')
    # print('Complex Accuracy:', trained_complex_all_digit_model.evaluate(x_test, y_test))
    # print('Simple Accuracy:', trained_simple_all_digit_model.evaluate(x_test, y_test))

def main():
    get_combined_histogram_data(simple_data=True)

if __name__ == '__main__':
    main()