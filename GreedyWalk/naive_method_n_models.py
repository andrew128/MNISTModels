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

import helpers.helper_funcs as helpers

def imagenet_postprocess(scores): 
    '''
    Source: https://github.com/onnx/models/blob/master/vision/classification/imagenet_postprocess.py

    Postprocessing with mxnet gluon
    The function takes scores generated by the network and returns the class IDs in decreasing order
    of probability
    '''
    prob = tf.nn.softmax(scores).numpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    return a

def imagenet_preprocess(img_data):
    '''
    Source: https://github.com/onnx/models/blob/master/vision/classification/imagenet_preprocess.py
    '''
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def mobilenet():
    return prepare(onnx.load('onnx_models/mobilenetv2-1.0/mobilenetv2-1.0.onnx'))

def resnet():
    return prepare(onnx.load('onnx_models/resnet18v1/resnet18v1.onnx'))

def squeezenet():
    return prepare(onnx.load('onnx_models/squeezenet1.1/squeezenet1.1.onnx'))

def vggnet():
    return prepare(onnx.load('onnx_models/vgg16/vgg16.onnx'))

def alexnet():
    return prepare(onnx.load('onnx_models/bvlc_alexnet/model.onnx'))

def run_model(tf_rep, test_data_dir='onnx_models/mobilenetv2-1.0/test_data_set_0'):
    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))

    # Run the model on the backend
    outputs = tf_rep.run(inputs)[0]


    np.save('model_prediction_not_processed', outputs)
    # outputs = imagenet_postprocess(outputs)
    # print(outputs[:10,:])
    print(outputs.shape)
    # print(outputs[0,0])
    print(outputs)
    # np.save('model_prediction', outputs)

    # Compare the results with reference outputs.
    # i = 0
    # for ref_o, o in zip(ref_outputs, outputs):
    #     print(i, ref_o.shape, o.shape)
    #     i = i + 1
        # print(np.allclose(ref_o, o, atol=1e-3, rtol=1e-3))
        # np.testing.assert_almost_equal(ref_o, o, decimal=5)

def get_sample_data():
    '''
    Combine all input_*.pb files and output_*.pb files into a 
    dictionary with key being the "model.test_dir_number_x" and 
    the value being a list of length 2. The first value is the 
    input and the second value is the output.
    '''
    dirToInputOutput = {}

    input_files = glob.glob('onnx_models/*/*/input_*.pb')
    for input_file in input_files:
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        parsed = numpy_helper.to_array(tensor)

        split_result = input_file.split('/')
        key = split_result[1] + '.' + split_result[2]

        dirToInputOutput[key] = [parsed]

    output_files = glob.glob('onnx_models/*/*/output_*.pb')
    for output_file in output_files:
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        parsed = numpy_helper.to_array(tensor)


        split_result = output_file.split('/')
        key = split_result[1] + '.' + split_result[2]

        if not key in dirToInputOutput.keys():
            print("ERROR: %v not in dictionary", output_file)
            break

        dirToInputOutput[key].append(parsed)

    return dirToInputOutput

def get_optimal_conf_values(models, time_constraint, conf_value_inputs, conf_value_labels):
    num_conf_vals = len(models) - 1

    conf_values = np.zeros(num_conf_vals)

    _, best_conf_values = get_optimal_conf_values_helper(models, conf_values, \
                            0, time_constraint, conf_value_inputs, conf_value_labels)

    return best_conf_values
   
def get_optimal_conf_values_helper(models, conf_values, curr_index, time_constraint,\
                                     conf_value_inputs, conf_value_labels):
    '''
    Return accuracy and array of optimal confidence values and None, None
    if no set of confidence values found that runs within the optimal time constraint.
    '''
    if curr_index == len(conf_values):
        # Run models with conf_values and return accuracy, none if doesn't satisfy time constraint
        # also return corresponding conf values
        accuracy, time = run_combined(models, conf_values, conf_value_inputs, conf_value_labels)
        if time < time_constraint:
            return accuracy, np.copy(conf_values) # TODO: think if there is any way to get rid of copy
        else:
            return None, None
    else:
        best_accuracy = 0
        best_conf_values = None

        for conf_value in np.arange(0, 1.1, 0.1):
            conf_values[curr_index] = conf_value
            accuracy, conf_values = get_optimal_conf_values(models, conf_values, \
                                                            curr_index + 1, time_constraint)
            if not accuracy == None and accuracy > best_accuracy:
                best_conf_values = conf_values
                best_accuracy = accuracy

        return best_accuracy, best_conf_values

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

def run_single(model, inputs):
    return model.run(inputs)[0]

def run_combined(models, conf_values, inputs, labels):
    assert len(models) - 1 == conf_values
    before_time = time.time()
    current_inputs_under_consideration = inputs

    combined_preds = np.arange(inputs.shape[0])

    for index, model in enumerate(models):
        # Separate out the inputs that current model will run and the 
        # inputs that the remaining group of models will consider.
        current_probs = run_single(model, current_inputs_under_consideration)
        current_conf_value = conf_values[index]

        current_highest_probs = np.amax(current_probs, axis=1)

        indices = [i for i in range(inputs.shape[0])]

        # Make predictions with current model and append to combined_predictions
        current_indices = np.where(current_highest_probs >= current_conf_value, indices, None)
        current_indices = current_indices[current_indices != np.array(None)] # remove None values
        current_indices = np.asarray(current_indices, dtype=np.int64)

        reduced_current_probs = np.take(current_probs, current_indices, axis=0)
        current_preds = reduced_current_probs.argmax(axis=1)

        np.put(combined_preds, current_indices, current_preds)

        # Deal with remaining model work
        remaining_indices = np.where(current_highest_probs < current_conf_value, indices, None)
        remaining_indices = remaining_indices[remaining_indices != np.array(None)] # remove None values
        remaining_indices = np.asarray(remaining_indices, dtype=np.int64)

        remaining_inputs = np.take(inputs, remaining_indices, axis=0)

        if remaining_inputs.shape[0] == 0:
            break
        else:
            current_inputs_under_consideration = \
                np.hstack((current_inputs_under_consideration, remaining_inputs))

    # Return accuracy of combined predictions and overall run time
    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels),\
         tf.float32)).numpy(), time.time() - before_time

def naive_search(models, validation_inputs, validation_labels,\
                 conf_value_inputs, conf_value_labels, input_time_constraint):
    '''
    Given list of onnx models, find combination of models that satisfied input time constraint

    Conf_value data is used to find the optimal confidence values for a set of models.

    Validation data used to validate optimal confidence value found - the accuracy and time
    of the validation data is used to determine the best model combination.
    '''

    # Enumerate all possible model combinations (n!) - use indices
    all_model_combinations = get_all_model_combinations(len(models))

    best_combination = None
    best_accuracy = 0

    # Loop through each model combination to find the optimal confidence value for each.
    for model_combination in all_model_combinations:
        if len(model_combination) == 0:
            continue
        input_models = np.take(model_combination, models)
        conf_values = get_optimal_conf_values(input_models, input_time_constraint,\
             conf_value_inputs, conf_value_labels)

        # Test model combination with optimal confidence values    
        accuracy, time = run_combined(input_models, conf_values, validation_inputs, validation_labels)
        if time <= input_time_constraint and accuracy > best_accuracy:
            best_combination = model_combination

    # Record best model combination
    return best_combination

def save_class_labels(synset_mapping_file):
    '''
    Saves the synset mapping as a npy file
    n* -> label
    '''
    idToLabel = {}
    with open(synset_mapping_file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            idToLabel[line.split(' ')[0]] = cnt
            cnt += 1

            line = fp.readline()

    np.save('synset_map', idToLabel)

def get_imagenet_validation_data(validation_data):
    '''
    Save map from filename (ILSVRC_2012_val_*) -> n* 
    Removes images with duplicates
    '''
    fileNameToID = {}
    with open(validation_data) as fp:
        line = fp.readline() # Read past the first line (header)
        line = fp.readline()
        cnt = 1
        while line:
            # print(line, '!!!')
            split_line = line.split(',')
            bounding_box_data = split_line[1].split(' ')
            # print(bounding_box_data)
            # Filter out images with multiple detectable objects
            if len(bounding_box_data) != 6:
                line = fp.readline()
                continue

            fileNameToID[split_line[0]] = bounding_box_data[0]

            line = fp.readline()
            if cnt < 10:
                print(cnt, split_line[0], bounding_box_data[0])
            # print(cnt)
            cnt += 1

    np.save('fileNameToID', fileNameToID)

def main():
    warnings.filterwarnings('ignore') # Ignore all the warning messages 

    # ---------------------
    # preprocessed_image = imagenet_preprocess('./ILSVRC2012_val_00011239.JPEG')
    preprocessed_image = np.load('x_val.npy').reshape(1, 3, 224, 224)
    print(preprocessed_image.shape)
    # print(preprocessed_image)
    # alexnet = alexnet()
    outputs = alexnet().run(preprocessed_image)[0]
    print(outputs)
    print(outputs.shape)

    # ---------------------

    # save_class_labels('LOC_synset_mapping.txt')
    # idToLabel = np.load('synset_map.npy', allow_pickle=True)
    # print(idToLabel)

    # get_imagenet_validation_data('LOC_val_solution.csv')
    # fileNameToID = np.load('fileNameToID.npy', allow_pickle=True)
    # print(fileNameToID)

    # output = np.load('model_prediction.npy')
    # unique = np.unique(output)
    # print(len(unique))
    # naive_search()

    # get_data()

    # run_model(alexnet())
    # run_model(vggnet())
    # run_model(squeezenet())
    # run_model(resnet())
    # run_model(mobilenet())

if __name__ == '__main__':
    main()
