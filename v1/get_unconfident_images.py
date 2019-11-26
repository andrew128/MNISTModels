import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import helper_funcs as helpers
from models import *

'''
This file visualizes inputs where the model is within a certain confidence threshold and wrong.
'''
def get_inputs_within_confidence_threshold(model, inputs, labels, prob, x, correct=False):
    '''
    Return list of images with highest predicted probability within x of prob and list of metadata
    associated with image containing tuples (highest probability, predicted label, actual label)
    '''
    prediction_probs = model.predict(inputs)
    predictions = prediction_probs.argmax(axis=1)
    highest_probs = np.amax(prediction_probs, axis=1)

    output_data = []
    output_metadata = []
    for i in range(predictions.shape[0]):
        if not correct and predictions[i] != labels[i]:
            if abs(highest_probs[i] - prob) <= x:
                output_data.append(inputs[i])
                output_metadata.append((highest_probs[i], predictions[i], labels[i]))

    return output_data, output_metadata

def main():
    x_train, y_train, x_test, y_test = helpers.get_data()
    trained_complex_all_digit_model = tf.keras.models.load_model('trained_complex_all_digit_model')

    incorrect_indecisive_data, incorrect_indecisive_metadata = get_inputs_within_confidence_threshold(trained_complex_all_digit_model, \
                                                                        x_test, y_test, 0.5, 0.05)

    print(len(incorrect_indecisive_data))
    helpers.visualize_mnist_data(incorrect_indecisive_data, incorrect_indecisive_metadata)
    # for i in range(5):
    #     plt.imshow(np.squeeze(x_train[i]), cmap='gray')
    #     plt.show()
        # print(np.squeeze(x_train[i]).shape)

if __name__ == '__main__':
    main()