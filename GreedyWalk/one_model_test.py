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

class OnnxModel():
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.input_shape = [1, 3, 224, 224]
        self.normalize = False

    def run_model(self, input_data):
        preprocessed_images = input_data.reshape(self.input_shape)
        scores = self.model.run(preprocessed_images)[0]
        probs = tf.nn.softmax(scores).numpy()
        return probs

def mobilenet():
    '''
    input: float[N, 3, 224, 224]
    output: scores
    '''
    return prepare(onnx.load('onnx_models/mobilenetv2-1.0/mobilenetv2-1.0.onnx'))

def resnet():
    '''
    input: float[N, 3, 224, 224]
    output: scores
    '''
    return prepare(onnx.load('onnx_models/resnet18v1/resnet18v1.onnx'))

def squeezenet():
    '''
    input: float[N, 3, 224, 224]
    output: scores
    '''
    return prepare(onnx.load('onnx_models/squeezenet1.1/squeezenet1.1.onnx'))

def vggnet():
    '''
    input: float[N, 3, 224, 224]
    output: scores
    '''
    return prepare(onnx.load('onnx_models/vgg16/vgg16.onnx'))

def alexnet():
    '''
    input: float[1, 3, 224, 224]
    output: float[1, 1000]
    '''
    return prepare(onnx.load('onnx_models/bvlc_alexnet/model.onnx'))

def run_model(model, model_name, inputs):
    output = model.run(inputs)[0]
    np.save('model_' + model_name, output, allow_pickle=True)

def main():
    warnings.filterwarnings('ignore') # Ignore all the warning messages 

    input_data_file = 'val_inputs_10000.npy'
    input_data = np.load(input_data_file)
    input_labels_file = 'val_labels_10000.npy'
    input_labels = np.load(input_labels_file)

    # Get indices where labels exist
    nonzero_indices = np.nonzero(input_labels[:, 0])
    nonzero_labels = input_labels[nonzero_indices]
    nonzero_inputs = input_data[nonzero_indices]
    # -----------------------------
    mobilenet_model = OnnxModel(mobilenet(), "mobilenet")

    output = mobilenet_model.run_model(nonzero_inputs)
    np.save('mobilenet_predictions.npy', output)
    print('mobilenet:')
    print(output.shape)
    print(np.sum(output))
    # -----------------------------
    resnet_model = OnnxModel(resnet(), "resnet")

    output = resnet_model.run_model(nonzero_inputs)
    np.save('mobilenet_predictions.npy', output)
    print('resnet:')
    print(output.shape)
    print(np.sum(output))
    # -----------------------------
    squeezenet_model = OnnxModel(squeezenet(), "squeezenet")

    output = squeezenet_model.run_model(nonzero_inputs)
    np.save('mobilenet_predictions.npy', output)
    print('squeezenet:')
    print(output.shape)
    print(np.sum(output))
    # -----------------------------
    vggnet_model = OnnxModel(vggnet(), "vggnet")

    output = vggnet_model.run_model(nonzero_inputs)
    np.save('mobilenet_predictions.npy', output)
    print('vggnet:')
    print(output.shape)
    print(np.sum(output))
    # -----------------------------
    alexnet_model = OnnxModel(alexnet(), "alexnet")

    output = alexnet_model.run_model(nonzero_inputs)
    np.save('mobilenet_predictions.npy', output)
    print('Alexnet:')
    print(output.shape)
    print(np.sum(output))

if __name__ == '__main__':
    main()