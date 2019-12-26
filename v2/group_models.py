import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from models import *
import helper_funcs as helpers 

def main():
    x_train, y_train, x_test, y_test = helpers.get_data()

    accuracies = []
    training_times = []
    testing_times = []

    for i in range(10):
        before_time = time.time()
        l1_model = get_trained_l1_all_digit_model(x_train, y_train)
        training_times.append(time.time() - before_time)

        before_time = time.time()
        accuracies.append(l1_model.evaluate(x_test, y_test)[1])
        testing_times.append(time.time() - before_time)

        l1_model.save('./models/l1_model_' + str(i))

    # print(l3_model.evaluate(x_test, y_test))

    # print(helpers.get_model_testing_times(l3_model, x_test, y_test, 1))
    print(accuracies)
    print(training_times)
    print(testing_times)

if __name__ == "__main__":
    main()