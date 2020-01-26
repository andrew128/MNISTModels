import tensorflow as tf
import time
import models
import helper_funcs

def main():
    # Get CIFAR data
    train_images, train_labels, test_images, test_labels = helper_funcs.get_cifar10_data()

    # Train and save models on CIFAR data
    input_shape = (32, 32, 3)
    num_runs = 10

    # Recording training times
    l1_mean_train_time = 0
    l2_mean_train_time = 0
    l3_mean_train_time = 0
    l4_mean_train_time = 0

    print('Recording training times...')
    for i in range(num_runs):
        print(i, '/', num_runs)
        t0 = time.time()
        l1_model = models.get_trained_l1_all_digit_model(train_images, train_labels, epochs=10)

        t1 = time.time()
        l2_model = models.get_trained_l2_all_digit_model(train_images, train_labels, input_shape, epochs=10)

        t2 = time.time()
        l3_model = models.get_trained_l3_all_digit_model(train_images, train_labels, input_shape, epochs=10)

        t3 = time.time()
        l4_model = models.get_trained_l4_all_digit_model(train_images, train_labels, input_shape, epochs=10)
        t4 = time.time()

        l1_mean_train_time += t1 - t0
        l2_mean_train_time += t2 - t1
        l3_mean_train_time += t3 - t2
        l4_mean_train_time += t4 - t3

        l1_model.save('models/l1_cifar10_model' + str(i))
        l2_model.save('models/l2_cifar10_model' + str(i))
        l3_model.save('models/l3_cifar10_model' + str(i))
        l4_model.save('models/l4_cifar10_model' + str(i))

    l1_mean_train_time /= num_runs
    l2_mean_train_time /= num_runs
    l3_mean_train_time /= num_runs
    l4_mean_train_time /= num_runs

    # Recording test times and accuracies

    l1_mean_test_time = 0
    l2_mean_test_time = 0
    l3_mean_test_time = 0
    l4_mean_test_time = 0

    l1_mean_accuracy = 0
    l2_mean_accuracy = 0
    l3_mean_accuracy = 0
    l4_mean_accuracy = 0

    print('Recording test times and accuracies...')
    for i in range(num_runs):
        print(i, '/', num_runs)
        l1_model = tf.keras.models.load_model('./models/l1_cifar10_model' + str(i))
        l2_model = tf.keras.models.load_model('./models/l2_cifar10_model' + str(i))
        l3_model = tf.keras.models.load_model('./models/l3_cifar10_model' + str(i))
        l4_model = tf.keras.models.load_model('./models/l4_cifar10_model' + str(i))

        t0 = time.time()
        l1_mean_accuracy += l1_model.evaluate(test_images, test_labels, verbose=0)[1]
        t1 = time.time()
        l2_mean_accuracy += l2_model.evaluate(test_images, test_labels, verbose=0)[1]
        t2 = time.time()
        l3_mean_accuracy += l3_model.evaluate(test_images, test_labels, verbose=0)[1]
        t3 = time.time()
        l4_mean_accuracy += l4_model.evaluate(test_images, test_labels, verbose=0)[1]
        t4 = time.time()

        l1_mean_test_time += t1 - t0
        l2_mean_test_time += t2 - t1
        l3_mean_test_time += t3 - t2
        l4_mean_test_time += t4 - t3

    l1_mean_test_time /= num_runs
    l2_mean_test_time /= num_runs
    l3_mean_test_time /= num_runs
    l4_mean_test_time /= num_runs

    l1_mean_accuracy /= num_runs
    l2_mean_accuracy /= num_runs
    l3_mean_accuracy /= num_runs
    l4_mean_accuracy /= num_runs

    print('Training Times')
    print(l1_mean_train_time)
    print(l2_mean_train_time)
    print(l3_mean_train_time)
    print(l4_mean_train_time)

    print('Testing times')
    print(l1_mean_test_time)
    print(l2_mean_test_time)
    print(l3_mean_test_time)
    print(l4_mean_test_time)

    print('Accuracies')
    print(l1_mean_accuracy)
    print(l2_mean_accuracy)
    print(l3_mean_accuracy)
    print(l4_mean_accuracy)

if __name__ == '__main__':
    main()