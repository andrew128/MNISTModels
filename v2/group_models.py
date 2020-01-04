import tensorflow as tf
import numpy as np
import time
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from models import *
import helper_funcs as helpers 

def print_model_data(x_train, y_train, x_test, y_test):
    '''
    Train and save models.
    '''
    accuracies = []
    training_times = []
    testing_times = []

    for i in range(10):
        before_time = time.time()
        l1_model = get_trained_l1_all_digit_model(x_train, y_train)
        training_times.append(time.time() - before_time)

        before_time = time.time()
        accuracies.append(l1_model.evaluate(x_test, y_test)[1])
        testing_times.append(time.time() - before_time)

        l1_model.save('./models/l1_model_' + str(i))

    # print(l3_model.evaluate(x_test, y_test))

    # print(helpers.get_model_testing_times(l3_model, x_test, y_test, 1))
    print(accuracies)
    print(training_times)
    print(testing_times)

def get_percentage_splits(increment=0.1):
    '''
    Return all of the percentage splits in groups of 3.
    '''
    splits = []
    i = 0
    while i <= 1:
        if i == 1:
            splits.append([1, 0, 0])
        else:
            bound = 1 - i
            j = 0
            while j <= bound:
                splits.append([i, j, 1 - i - j])
                j += increment

        i += increment

    # print(splits)
    return splits

    # splits = []

    # l1_splits = np.arange(0, 1 + increment, increment)
    # for split_1 in l1_splits:
    #     l2_splits = np.arange(0, 1, increment)
    #     # print(split_1, splits_2)
    #     for split_2 in l2_splits:
    #         # print(split_1, split_2, 1 - split_1 - split_2)
    #         splits.append([split_1, split_2, 1 - split_1 - split_2])
    
    # # splits.append([1, 0, 0])
    # print(splits)
    # return splits

def get_accuracies_and_times(splits):
    x_train, y_train, x_test, y_test = helpers.get_data()

    l1_model = tf.keras.models.load_model('./models/l1_model_0')
    l2_model = tf.keras.models.load_model('./models/l2_model_0')
    l3_model = tf.keras.models.load_model('./models/l3_model_0')

    accuracies = []
    times = []

    num_epochs = 10

    for split in splits:
        accuracy_sum = 0
        time_sum = 0
        for _ in range(num_epochs):
            before_time = time.time()

            l1_split = split[0]
            l2_split = split[1]
            l3_split = split[2]
            print(split)

            # Shuffle testing inputs and labels (so each split won't 
            # always be evaluated on the same data)
            indices = tf.random.shuffle(tf.range(y_test.shape[0]))
            x_test = tf.gather(x_test, indices)
            y_test = tf.gather(y_test, indices)

            # Split testing inputs and labels
            s_1 = int(l1_split * y_test.shape[0])
            s_2 = int(l2_split * y_test.shape[0] + s_1)

            x_test_1 = x_test[0:s_1]
            x_test_2 = x_test[s_1:s_2]
            x_test_3 = x_test[s_2:y_test.shape[0]]

            y_test_1 = y_test[0:s_1]
            y_test_2 = y_test[s_1:s_2]
            y_test_3 = y_test[s_2:y_test.shape[0]]

            # Evaluate and combine accuracies
            l1_accuracy = 0
            l2_accuracy = 0
            l3_accuracy = 0

            if x_test_1.shape[0] > 0:
                l1_accuracy = l1_model.evaluate(x_test_1, y_test_1, verbose=0)[1]
            if x_test_2.shape[0] > 0:
                l2_accuracy = l2_model.evaluate(x_test_2, y_test_2, verbose=0)[1]
            if x_test_3.shape[0] > 0:
                l3_accuracy = l3_model.evaluate(x_test_3, y_test_3, verbose=0)[1]

            total_accuracy = l1_split * l1_accuracy + l2_split * l2_accuracy + l3_split * l3_accuracy

            time_sum += time.time() - before_time
            accuracy_sum += total_accuracy

        times.append(time_sum / num_epochs)
        accuracies.append(total_accuracy / num_epochs)
        # times.append(time.time() - before_time)
        # accuracies.append(total_accuracy)
    
    return accuracies, times

def view_3d(splits, z, accuracies=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # x, y, z = axes3d.get_test_data(0.05)
    splits = np.asarray(splits)
    print(splits.shape)
    x = splits[:,0]
    y = splits[:,1]
    # z = accuracies
    # x = [1, 4]
    # y = [2, 3]
    # z = np.array([[3, 4]])

    ax.set_ylim(ax.get_ylim()[::-1])
    if accuracies:
        ax.set_title('Accuracy vs Split Percentages')
    else:
        ax.set_title('Times vs Split Percentages')

    ax.set_xlabel('L1 Percentage')
    ax.set_ylabel('L2 Percentage')

    if accuracies:
        ax.set_zlabel('Accuracy (%)')
    else:
        ax.set_zlabel('Time (sec)')
    ax.plot_trisurf(x,y,z)
    # ax.plot_surface(x,y,z)
    # ax.plot_wireframe(x,y,z)

    plt.show()

def main():
    # print('Calculating splits...')
    splits = get_percentage_splits(increment=0.1)
    # print('Calculating accuracies...')
    accuracies, times = get_accuracies_and_times(splits)
    np.save('avg_accuracies_0.1', accuracies)
    np.save('avg_times_0.1', times)

    # accuracies = np.load('accuracies_0.1.npy')
    # times = np.load('times_0.1.npy')
    print(times) 
    view_3d(splits, accuracies)
    view_3d(splits, times, accuracies=False)

if __name__ == "__main__":
    main()