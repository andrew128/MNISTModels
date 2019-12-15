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
        self.complex_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
        self.simple_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')

    def predict_probs_time(self, inputs, max_time):
        pass

    def predict_probs_conf(self, inputs, confidence_bound):
        # Make all simple predictions
        simple_probs = self.simple_model.predict(inputs)
        simple_highest_probs = np.amax(simple_probs, axis=1)
        simple_preds = simple_probs.argmax(axis=1)

        combined_predictions = tf.cond(simple_highest_probs > confidence_bound, lambda: simple_preds, lambda: self.complex_model.predict(inputs).argmax(axis=1))
        # combined_predictions = np.where(simple_highest_probs > confidence_bound, simple_preds, complex_preds)

    def evaluate(self):
        pass

def cond(x):
    print('cond', x)

def body(x):
    print('body', x)

def main():
    # x_train, y_train, x_test, y_test = helpers.get_data()
    # model = CombinedModel()
    # x_test_subset = x_test[1001:1006]
    # y_test_subset = y_test[1001:1006]
    # print('Labels:', y_test_subset)
    # simple_preds = model.simple_model.predict(x_test).argmax(axis=1)[1001:1006]
    # print('Simple preds:', simple_preds)
    # complex_preds = model.complex_model.predict(x_test).argmax(axis=1)[1001:1006]
    # print('Complex preds:', complex_preds)
    # ================================
    conf_val = 0.5
    simple_probs = [0.9, 0.91, 0.8, 0.1, 0.2]
    simple_preds = [5, 6, 7, 8, 9]
    complex_preds = [0, 1, 2, 3, 4]

    # tf.map_fn(lambda x: tf.cond(), simple_probs)
    x = tf.Variable(tf.constant(0, shape=[2, 2]))
    tf.while_loop(cond, body, [simple_probs])


if __name__ == '__main__':
    main()