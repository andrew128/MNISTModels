import numpy as np
import onnx
import os
import time
import glob
import warnings
from copy import deepcopy

from onnx_tf.backend import prepare
from onnx import numpy_helper

import tensorflow as tf

from onnx_models import *

def get_all_model_combinations(n):
    '''
    Output is a list of lists.
    '''
    output_list = []
    get_all_model_combinations_helper(n, [], 0, output_list)
    return output_list

def get_all_model_combinations_helper(n, current_list, i, output_list):
    '''
    Return void
    '''
    if i == n:
        output_list.append(deepcopy(current_list))
    else:
        get_all_model_combinations_helper(n, current_list, i + 1, output_list)
        current_list.append(i)
        get_all_model_combinations_helper(n, current_list, i + 1, output_list)
        current_list.remove(i)

def run_pair(model_1, model_2, conf_value, inputs, labels):
    '''
    model 1 is simple
    model 2 is complex

    output: (accuracy, time)
    '''
    simple_model = model_1
    complex_model = model_2

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

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy()

def run_all_possible_conf_values(combination, models, conf_value, inputs, labels):
    conf_values = np.arange(0, 1.1, 0.1)
    # Store combination, time, accuracy, conf_value by writing to file
    

def main():
    # Ignore all the warning messages
    warnings.filterwarnings('ignore') 

    # -----------------------------------
    input_data_file = 'val_inputs_10000.npy'
    input_data = np.load(input_data_file)
    input_labels_file = 'val_labels_10000.npy'
    input_labels = np.load(input_labels_file)

    # Get indices where labels exist
    nonzero_indices = np.nonzero(input_labels[:, 0])
    nonzero_labels = input_labels[nonzero_indices]
    nonzero_inputs = input_data[nonzero_indices]
    # -----------------------------------

    print('Construct mobilenet...')
    mobilenet_model = OnnxModel(mobilenet(), "mobilenet")
    print('Construct resnet...')
    resnet_model = OnnxModel(resnet(), "resnet")
    print('Construct squeezenet...')
    squeezenet_model = OnnxModel(squeezenet(), "squeezenet")
    # print('Construct vggnet...')
    # vggnet_model = OnnxModel(vggnet(), "vggnet")
    print('Construct alexnet...')
    alexnet_model = OnnxModel(alexnet(), "alexnet")

    # print('Creating map of indices to models...')
    # Create map of 0 -> 4 for each of 5 models
    num_models = 4
    map_indices_to_model = []
    map_indices_to_model.append(squeezenet_model)
    print('Appended squeeze net...')
    map_indices_to_model.append(mobilenet_model)
    print('Appended mobile net...')
    map_indices_to_model.append(resnet_model)
    print('Appended res net...')
    # map_indicies_to_model.append(vggnet_model)
    # print('Appended vgg net...')
    map_indices_to_model.append(alexnet_model)
    print('Appended alex net...')

    print('Generating possible pair combinations...')
    # Generate all possible combinations of 2 between 0 -> 4, sort
    # them, and put them into a set
    all_model_combinations = get_all_model_combinations(num_models)
    model_combinations_size_two = []
    for combination in all_model_combinations:
        if (len(combination) == 2):
            model_combinations_size_two.append(combination)

    # For each of the combinations, find the optimal confidence values
    # by looping through all possible
    for combination in model_combinations_size_two:
        # model_1 = map_indices_to_model[combination[0]]
        # model_2 = map_indices_to_model[combination[1]]
        # accuracy, time = run_pair

if __name__ == '__main__':
    main()