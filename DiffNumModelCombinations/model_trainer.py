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
    # l4_model = models.get_trained_l4_all_digit_model(x_train, y_train, input_shape)
    # l4_model.save('models/l4_model')

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    train_and_save_models(x_train, y_train)

if __name__ == '__main__':
    main()