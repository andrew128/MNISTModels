import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
import time

import helper_funcs as helpers
from models import *

class CombinedModel():
    def __init__(self):
        self.trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
        self.trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')

        self.lower_time_bound = 0.36
        self.time_conf_val_trendline_coef = [1.27, -0.624, 0.342]

    def get_best_potential_accuracy(self, confidence_value):
        if confidence_value > 1:
            return 0.9737
        return 0.915 - 0.0884 * confidence_value + 0.375 * (confidence_value**2)\
            - 0.23 * (confidence_value**3)

    def predict_probs_time(self, inputs, max_time):
        '''
        Max time should be in terms of seconds.
        '''
        if max_time < self.lower_time_bound:
            print('Error: max_time given below lowest possible time')
            return None

        deep_copy_coef = np.copy(self.time_conf_val_trendline_coef)
        deep_copy_coef[deep_copy_coef.shape[0] - 1] -= max_time
        confidence_value = np.amax(np.roots(deep_copy_coef).real)

        print('Confidence Value:', confidence_value)
        print('Best Potential Accuracy:', self.get_best_potential_accuracy(confidence_value))

        return self.predict_probs_conf(inputs, confidence_value)

    def predict_probs_conf(self, inputs, confidence_bound):
        # -----------------------------------
        # All Simple
        simple_probs = self.trained_simple_all_digit_model.predict(inputs)
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
            complex_preds = np.argmax(self.trained_complex_all_digit_model.predict(complex_inputs), axis=1)
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

        return combined_preds

    def evaluate_time(self, inputs, labels, max_time):
        if max_time < self.lower_time_bound:
            print('Error: max_time given below lowest possible time')
            return None
        preds = self.predict_probs_time(inputs, max_time)
        return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()

    def evaluate_conf(self, inputs, labels, confidence_bound):
        preds = self.predict_probs_conf(inputs, confidence_bound)
        return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()

def main():
    x_train, y_train, x_test, y_test = helpers.get_data()
    model = CombinedModel()

    times = np.arange(0.26, 1, 0.1)
    for single_time in times:
        before_time = time.time()
        accuracy = model.evaluate_time(x_test, y_test, single_time)
        duration = time.time() - before_time
        print('Bound:', single_time, 'Duration:', duration, 'Accuracy:', accuracy)

if __name__ == '__main__':
    main()