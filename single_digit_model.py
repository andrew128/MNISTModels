import tensorflow as tf
import numpy as np

class KerasModel(tf.keras.Model):
    def __init__(self,):
        """
        The model class inherits from tf.keras.Model.
        It stores the trainable weights as attributes.
        """
        super(KerasModel, self).__init__()
        # Instead of defining variables, we define layers:
        # when using Keras, we can do the linear layer and the activation function in one step
        self.dense_1 = tf.keras.layers.Dense(784, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        """
        Forward pass, predicts labels given an input image using fully connected layers
        :return: the probabilites of each label
        """
        L1output = self.dense_1(inputs)
        prbs = self.dense_2(L1output)

        return prbs

    def loss(self, predictions, labels):
        """
        Calculates the model loss
        :return: the loss of the model as a tensor
        """
        return tf.reduce_sum(tf.keras.losses.categorical_crossentropy(predictions, labels))  

    def accuracy(self, predictions, labels):
        """
        Calculates the model accuracy
        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(predictions, 1),
                        tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def train(keras_model, x_train, y_train):
    # Choosing an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Loop through 10000 training images
    for i in range(10000):
        image = np.reshape(x_train[i], (1,-1))
        label = y_train[i]

        # Implement backprop:
        with tf.GradientTape() as tape:
            predictions = keras_model.call(image) # call the call function
            loss = keras_model.loss(predictions, label) # call the loss function

            if i % 500 == 0:
                train_acc = keras_model.accuracy(keras_model(x_train.reshape(-1,784)), y_train)
                print("Accuracy on training set after {} training steps: {}".format(i, train_acc))

        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment, 
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    return model.accuracy(model(test_inputs.reshape(-1,784)), test_labels)

def get_data(digit_class, x_train, y_train):
    result_inputs = []
    result_labels = []
    for i in range(y_train.shape[0]):
        if y_train[i] == digit_class:
            result_inputs.append(x_train[i])
            result_labels.append(0) # First class represented as 0
        else:
            result_inputs.append(x_train[i])
            result_labels.append(1) # Second class represented as 1

    # Convert data to numpy arrays.
    result_inputs = np.asarray(result_inputs)
    result_labels = np.asarray(result_labels)

    # Normalize inputs
    result_inputs = result_inputs / 255

    # Turn labels into one-hot vectors.
    result_labels = tf.one_hot(result_labels, depth=2)

    # Change inputs from float64 to float32
    result_inputs = result_inputs.astype('float32')

    return (result_inputs, result_labels)


def main():
    # Loading in and preprocessing the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for i in range(10):
        result_inputs, result_labels = get_data(i, x_train, y_train)
        test_inputs, test_labels = get_data(i, x_test, y_test)

        model = KerasModel()
        train(model, result_inputs, result_labels)
        print(test(model, test_inputs, test_labels))
    

if __name__ == '__main__':
    main()