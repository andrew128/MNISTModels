import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time

class KerasModel(tf.keras.Model):
    def __init__(self,):
        super(KerasModel, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(784, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(2, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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

# def train(keras_model, x_train, y_train):
#     for i in range(x_train.shape[0]):
#         image = np.reshape(x_train[i], (1,-1))
#         label = y_train[i]

#         # Implement backprop:
#         with tf.GradientTape() as tape:
#             predictions = keras_model.call(image)
#             loss = keras_model.loss(predictions, label)

#             if i % 500 == 0:
#                 train_acc = keras_model.accuracy(keras_model(x_train.reshape(-1,784)), y_train)
#                 # print("Accuracy on training set after {} training steps: {}".format(i, train_acc))
#         gradients = tape.gradient(loss, keras_model.trainable_variables)
#         keras_model.optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

# def test_single_model(model, test_inputs, test_labels):
#     """
#     :param test_inputs: MNIST test data (all images to be tested)
#     :param test_labels: MNIST test labels (all corresponding labels)
#     :return: accuracy - Float (0,1)
#     """
#     return model.accuracy(model(test_inputs.reshape(-1,784)), test_labels)

# def test_ten_models(models, test_inputs, test_labels):
#     num_correct = 0
#     for input_image, input_label in zip(test_inputs, test_labels):
#         input_image = input_image.reshape(-1, 784)
#         all_probs = []
#         for model in models:
#             probabilities = model(test_inputs.reshape(-1,784))
#             all_probs.append(max(probabilities))
#         predicted_label = np.argmax(all_probs)
#         if predicted_label == input_label:
#             num_correct += 1;
#     return num_correct / test_labels.shape[0]

def get_model():
    input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(Dense(2,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model    

def get_data(digit_class, inputs, labels):
    '''
    Modify training data to be classified either as the input digit_class or not.
    '''

    # Reshaping the array to 4-dims so that it can work with the Keras API
    inputs = inputs.reshape(inputs.shape[0], 28, 28, 1)

    result_inputs = []
    result_labels = []
    for i in range(labels.shape[0]):
        if labels[i] == digit_class:
            result_inputs.append(inputs[i])
            result_labels.append(0) # First class represented as 0
        else:
            result_inputs.append(inputs[i])
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
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train, y_train = get_data(0, x_train, y_train)
    # x_test, y_test = get_data(0, x_test, y_test)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train_4 = (y_train == 4)#True for all 4s, False for all other digits
    y_test_4 = (y_test == 4)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = get_model()
    model.fit(x=x_train,y=y_train, epochs=1)
    model.evaluate(x_test, y_test)
    # probabilities = model(x_test.reshape(-1,784))
    # print(probabilities)

    # models = []
    # for i in range(10):    
    #     result_inputs, result_labels = get_data(i, x_train, y_train)
    #     test_inputs, test_labels = get_data(i, x_test, y_test)
    #     model = KerasModel()
    #     train(model, result_inputs, result_labels)
    #     print('Model', i, 'accuracy', test_single_model(model, test_inputs, test_labels))
    #     models.append(model)

    # test_ten_models(models, x_test, y_test)

    # for model in models:
    #     print(test_single_model(model, test_inputs, test_labels))
    

if __name__ == '__main__':
    main()