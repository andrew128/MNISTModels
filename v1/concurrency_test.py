import multiprocessing
import numpy as np
import time
import logging
from multiprocessing.util import log_to_stderr
from os import getpid

import helper_funcs as helpers

x_train, y_train, x_test, y_test = helpers.get_data()
confidence_bound = 0.56

def get_pred(index):
    print(getpid())
    import tensorflow as tf

    trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')
    trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')
    label = y_test[index]
    x = np.expand_dims(x_test[index], axis=0)
    probs = trained_simple_all_digit_model.predict(x)
    if np.amax(probs) > confidence_bound:
        return probs.argmax(axis=1)[0]
    else:
        return trained_complex_all_digit_model.predict(np.expand_dims(x_test[index], axis=0)).argmax(axis=1)[0]

def get_complex_pred():
    import tensorflow as tf

    trained_complex_all_digit_model = tf.keras.models.load_model('./models/trained_complex_all_digit_model')

    return trained_complex_all_digit_model.predict(x_test).argmax(axis=1)[0]

def get_simple_pred():
    import tensorflow as tf

    trained_simple_all_digit_model = tf.keras.models.load_model('./models/trained_simple_all_digit_model')

    return trained_simple_all_digit_model.predict(x_test).argmax(axis=1)[0]

if __name__ == "__main__":
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        indices = [i for i in range(y_test.shape[0])]

        indices = [i for i in range(4)]
        combined_results = pool.map(get_pred, indices)

    # get_complex_pred()
    # get_simple_pred()

    duration = time.time() - start_time

    # print(f"Accuracy {accuracy}")
    print(f"Duration {duration} seconds")