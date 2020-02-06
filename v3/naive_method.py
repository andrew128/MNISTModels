import tensorflow as tf
import numpy as np
import time
import helper_funcs
import models

def get_naive_method_times_and_accuracies(model_1, model_2, x_test, y_test, splits):
    '''
    Prints arrays where each index contains the times and accuracies of the combined model 
    for a list of data splits.
    Combined model is made up of 2 input models.

    Time and accuracy for split i is i model_1, 1 - i model_2.
    Time and accuracy for each split is the mean over 10 epochs.

    model_2 is more complex then model_1.
    '''

    y_test = tf.squeeze(y_test)

    combined_times = []
    combined_accuracies = []

    # For each test split, run combined predictions.
    for data_split in splits:
        print('Data Split:', data_split)

        total_combined_time = 0
        total_combined_accuracy = 0

        num_epochs = 5
        for i in range(num_epochs):
            # print(' Epoch:', i)
            # Shuffle data
            indices = tf.random.shuffle(tf.range(y_test.shape[0]))
            x_test = tf.gather(x_test, indices)
            y_test = tf.gather(y_test, indices)

            # Split data
            split_index = tf.cast(data_split * y_test.shape[0], tf.int32)

            # print('Split index', split_index)
            simple_x = x_test[:split_index]
            simple_y = y_test[:split_index]

            complex_x = x_test[split_index:]
            complex_y = y_test[split_index:]

            # Combined
            before_time = time.time()

            # Complex portion
            if complex_x.shape[0] != 0:
                complex_probs = model_2.predict(complex_x)
                complex_preds = complex_probs.argmax(axis=1)
                complex_output = tf.equal(complex_preds, complex_y)

            # Simple portion
            if simple_x.shape[0] != 0:
                simple_probs = model_1.predict(simple_x)
                simple_preds = simple_probs.argmax(axis=1)
                simple_output = tf.equal(simple_preds, simple_y)

            # Combined
            if simple_x.shape[0] != 0 and complex_x.shape[0] != 0:
                combined_output = tf.concat([complex_output, simple_output], axis=0)
            elif simple_x.shape[0] != 0:
                combined_output = simple_output
            else:
                combined_output = complex_output
            combined_accuracy = tf.reduce_mean(tf.cast(combined_output, tf.float32))

            after_time = time.time()
            # print(after_time - before_time, combined_accuracy)
            total_combined_time += after_time - before_time
            total_combined_accuracy += combined_accuracy

        combined_times.append(total_combined_time / num_epochs)
        combined_accuracies.append(total_combined_accuracy.numpy() / num_epochs)

    print('Combined Times:', combined_times)
    print('Combined Accuracies:', combined_accuracies)

def main():
    # Get CIFAR data
    train_images, train_labels, test_images, test_labels = helper_funcs.get_cifar10_data()

    # Train and save models on CIFAR data
    # input_shape = (32, 32, 3)
    # l1_model = models.get_trained_l1_all_digit_model(train_images, train_labels, epochs=10)
    # l2_model = models.get_trained_l2_all_digit_model(train_images, train_labels, input_shape, epochs=10)
    # l3_model = models.get_trained_l3_all_digit_model(train_images, train_labels, input_shape, epochs=10)
    # l4_model = models.get_trained_l4_all_digit_model(train_images, train_labels, input_shape, epochs=10)

    # l1_model.save('models/l1_cifar10_model')
    # l2_model.save('models/l2_cifar10_model')
    # l3_model.save('models/l3_cifar10_model')
    # l4_model.save('models/l4_cifar10_model')

    l1_model = tf.keras.models.load_model('./models/l1_cifar10_model0')
    l2_model = tf.keras.models.load_model('./models/l2_cifar10_model0')
    l3_model = tf.keras.models.load_model('./models/l3_cifar10_model0')
    l4_model = tf.keras.models.load_model('./models/l4_cifar10_model0')
    # print(l4_model.summary())

    # print(l1_model.metrics_names)
    # print(l2_model.metrics_names)
    # print(l3_model.metrics_names)
    # print(l4_model.metrics_names)

    # print(l1_model.evaluate(test_images, test_labels, verbose=0))
    # print(l2_model.evaluate(test_images, test_labels, verbose=0))
    # print(l3_model.evaluate(test_images, test_labels, verbose=0))
    # print(l4_model.evaluate(test_images, test_labels, verbose=0))

    # Get and save inference time and accuracies on test data vs splits data
    splits = np.arange(0, 1.1, 0.1)
    print('L1 and L2')
    get_naive_method_times_and_accuracies(l1_model, l2_model, test_images, test_labels, splits)
    print('L1 and L3')
    get_naive_method_times_and_accuracies(l1_model, l3_model, test_images, test_labels, splits)
    print('L1 and L4')
    get_naive_method_times_and_accuracies(l1_model, l4_model, test_images, test_labels, splits)
    print('L2 and L3')
    get_naive_method_times_and_accuracies(l2_model, l3_model, test_images, test_labels, splits)
    print('L2 and L4')
    get_naive_method_times_and_accuracies(l2_model, l4_model, test_images, test_labels, splits)
    print('L3 and L4')
    get_naive_method_times_and_accuracies(l3_model, l4_model, test_images, test_labels, splits)

if __name__ == '__main__':
    main()