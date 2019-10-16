import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
  def __init__(self,):
    """
    The model class inherits from tf.keras.Model.
    It stores the trainable weights as attributes.
    """
    super(Model, self).__init__()
    
    # Initialize your variables (weights) here:
    # Think about using the gaussian_initialization example for these variables
    # Remember that the input size is 784 and the output size (number of classes) is 10 
    self.W1 = tf.Variable(tf.random.normal(shape=[784,100], stddev=.1, dtype=tf.float32))
    self.b1 = tf.Variable(tf.random.normal(shape=[100,], stddev=.1, dtype=tf.float32))
    self.W2 = tf.Variable(tf.random.normal(shape=[100, 10], stddev=.1, dtype=tf.float32))
    self.b2 = tf.Variable(tf.random.normal([10], stddev=.1, dtype=tf.float32))
  
  def call(self, inputs):
    """
    Forward pass, predicts labels given an input image using fully connected layers
    :return: the probabilites of each label
    """
    layer1Output = tf.nn.relu(tf.matmul(inputs, self.W1) + self.b1) # remember to use a relu activation
    logits = tf.matmul(layer1Output, self.W2) + self.b2
    prbs = tf.nn.softmax(logits)
    
    return prbs
  
  def loss(self, predictions, labels):
    """
    Calculates the model loss
    :return: the loss of the model as a tensor
    """
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(tf.clip_by_value(predictions,1e-10,1.0)),axis=[1]))
  
  def accuracy(self, predictions, labels):
    """
    Calculates the model accuracy
    :return: the accuracy of the model as a tensor
    """

    correct_prediction = tf.equal(tf.argmax(predictions, 1),
                    tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def main():
    # Loading in and preprocessing the data
    mnist = tf.keras.datasets.mnist
    # x_train is your train inputs
    # y_train is your train labels
    # x_test is your test inputs
    # y_test is your test labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 # normalizing data
    x_train = x_train.astype(np.float32)
    pass

if __name__ == '__main__':
    main()