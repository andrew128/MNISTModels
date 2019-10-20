import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
	def __init__(self):
		"""
    	This model class will contain the architecture for your CNN that 
		classifies images. Do not modify the constructor, as doing so 
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()

		self.batch_size = 64
		self.num_classes = 2
		self.hidden_layer1_size = 64
		self.hidden_layer2_size = 32
		self.epsilon = 1e-3

		# TODO: Initialize all trainable parameters
		self.conv1_filters = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], stddev=0.1))
		self.conv1_bias = tf.Variable(tf.random.truncated_normal([16])) # lec 11-45

		self.dense1_size = 4 * 4 * 20
		self.dense1_w = tf.Variable(tf.random.truncated_normal(\
			shape=[self.dense1_size,self.hidden_layer1_size],stddev=0.1),dtype=tf.float32)
		self.dense2_w = tf.Variable(tf.random.truncated_normal(\
			shape=[self.hidden_layer1_size,self.hidden_layer2_size],stddev=0.1),dtype=tf.float32)
		self.dense3_w = tf.Variable(tf.random.truncated_normal(\
			shape=[self.hidden_layer2_size,self.num_classes],stddev=0.1),dtype=tf.float32)

		self.dense1_b = tf.Variable(tf.random.truncated_normal(\
			shape=[self.hidden_layer1_size],stddev=0.1),dtype=tf.float32)
		self.dense2_b = tf.Variable(tf.random.truncated_normal(\
			shape=[self.hidden_layer2_size],stddev=0.1),dtype=tf.float32)
		self.dense3_b = tf.Variable(tf.random.truncated_normal(\
			shape=[self.num_classes],stddev=0.1),dtype=tf.float32)

	def call(self, inputs, is_testing=False):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
		# shape of filter = (filter_height, filter_width, in_channels, out_channels)
		# shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

		# Layer 1::(batch_size, 32, 32, 3) -> (batch_size, 8, 8, 16)
		conv1 = tf.nn.conv2d(inputs, self.conv1_filters, (1, 2, 2, 1), padding="SAME")
		# print('conv1 shape ', conv1.shape)
		conv1_with_bias = tf.nn.bias_add(conv1, self.conv1_bias)
		batch1_mean_1, batch1_var_1 = tf.nn.moments(conv1_with_bias, [0, 1, 2]) # Gets mean and variance
		batch1_norm_1 = tf.nn.batch_normalization(conv1_with_bias, batch1_mean_1, batch1_var_1, None, None, self.epsilon)
		relu1 = tf.nn.relu(batch1_norm_1)
		pooled_conv1 = tf.nn.max_pool(relu1, ksize=(3,3), strides=(2, 2), padding="SAME")

		# print('pool1 output shape ', pooled_conv1.shape)

		# Layer 2::(batch_size, 8, 8, 16) -> (batch_size, 4, 4, 20)
		conv2 = tf.nn.conv2d(pooled_conv1, self.conv2_filters, (1, 1, 1, 1), padding="SAME")
		conv2_with_bias = tf.nn.bias_add(conv2, self.conv2_bias)
		batch2_mean_1, batch2_var_1 = tf.nn.moments(conv2_with_bias, [0, 1, 2]) # Gets mean and variance
		batch2_norm_1 = tf.nn.batch_normalization(conv2_with_bias, batch2_mean_1, batch2_var_1, None, None, self.epsilon)
		relu2 = tf.nn.relu(batch2_norm_1)
		pooled_conv2 = tf.nn.max_pool(relu2, ksize=(2,2), strides=(2, 2), padding="SAME")

		# print('pool2 output shape ', pooled_conv2.shape)

		# Layer 3::(batch_size, 4, 4, 20) -> (batch_size, 4, 4, 20)
		conv3 = None
		if is_testing == True:
			conv3 = conv2d(pooled_conv2, self.conv3_filters, (1, 1, 1, 1), padding="SAME")
		else:
			conv3 = tf.nn.conv2d(pooled_conv2, self.conv3_filters, (1, 1, 1, 1), padding="SAME")
		conv3_with_bias = tf.nn.bias_add(conv3, self.conv3_bias)
		batch3_mean_1, batch3_var_1 = tf.nn.moments(conv3_with_bias, [0, 1, 2]) # Gets mean and variance
		batch3_norm_1 = tf.nn.batch_normalization(conv3_with_bias, batch3_mean_1, batch3_var_1, None, None, self.epsilon)
		relu3 = tf.nn.relu(batch3_norm_1)

		# Dense Layers::(batch_size, 4, 4, 20) -> (batch_size, 2)
		# print(relu3.shape)
		dense1_input = tf.reshape(relu3, [-1, self.dense1_size])
		# print(dense1_input.shape)
		layer1_output = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(\
			tf.matmul(dense1_input, self.dense1_w), self.dense1_b)), 0.3)
		layer2_output = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(\
			tf.matmul(layer1_output, self.dense2_w), self.dense2_b)), 0.3)
		layer3_output = tf.nn.relu(tf.nn.bias_add(\
			tf.matmul(layer2_output, self.dense3_w), self.dense3_b))

		return layer3_output


	def loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes) 
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""
		# print(labels.shape, logits.shape)
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

		NOTE: DO NOT EDIT
		
		:return: the accuracy of the model as a Tensor
		"""
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training), 
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training), 
	shape (num_labels, num_classes)
	:return: None
	'''
	# Shuffle inputs and labels the same way.
	indices = tf.random.shuffle(tf.range(train_labels.shape[0]))
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

	train_inputs = tf.image.random_flip_left_right(train_inputs)

	# Loop through inputs in batches.
	for i in range(0, train_labels.shape[0], model.batch_size):

		curr_batch_inputs = train_inputs[i:i + model.batch_size]
		curr_batch_labels = train_labels[i:i + model.batch_size]

		with tf.GradientTape() as tape:
			predictions = model.call(curr_batch_inputs)
			loss = model.loss(predictions, curr_batch_labels)
		
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly 
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested), 
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across 
	all batches or the sum as long as you eventually divide it by batch_size
	"""

	sumAccuracy = 0
	numBatches = test_labels.shape[0] / model.batch_size

	print(test_labels.shape[0], model.batch_size)
	for i in range(0, test_labels.shape[0], model.batch_size):
		curr_batch_inputs = test_inputs[i:i + model.batch_size]
		curr_batch_labels = test_labels[i:i + model.batch_size]
		print(i)
		logits = model.call(curr_batch_inputs, is_testing=True)

		sumAccuracy = sumAccuracy + model.accuracy(logits, curr_batch_labels)

	# Returns average accuracy across all batches
	return sumAccuracy / numBatches
def main():
    # Loading in and preprocessing the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    x_train, x_test = x_train / 255.0, x_test / 255.0 # normalizing data
    x_train = x_train.astype(np.float32)

    model = KerasModel()

    train(model, x_train, y_train)

    x_test = x_test.astype(np.float32)
    print(test(model, x_test, y_test))
    

if __name__ == '__main__':
    main()