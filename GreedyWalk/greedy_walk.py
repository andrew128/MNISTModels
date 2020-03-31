import numpy as np
import time
import tensorflow as tf

import graph as Graph
import helpers.helper_funcs as helpers
import helpers.models as models

def greedy_walk(graph, search_time_constraint, model_time_constraint, \
                validation_inputs, validation_labels, conf_value_inputs,\
                conf_value_labels, epsilon = 0.1):
    '''
    :param: graph - Graph object representing all possible model pairings
    :param: search_time_constraint - amount of time given to search
    :param: model_time_constraint - amount of time output model pairing must 
                                    complete validation_dataset within
    :param: validation_inputs - inputs given to test each model pairing using
                               optimal confidence value
    :param: validation_labels - corresponding labels
    :param: conf_value_inputs - dataset given to find optimal confidence value
    :param: conf_value_labels - corresponding labels
    output: tuple of models of size 2

    This method finds the model pairing chosen from the input model set
    that achieves the highest accuracy on the input dataset within the model 
    pairing input time constraint. The result is returned within the search input
    time constraint.
    '''
    timeout_start = time.time()

    current_node = graph.get_most_simple_node()

    best_node_so_far = None
    best_accuracy_so_far = 0

    while time.time() < timeout_start + search_time_constraint:
        print('=========================================')
        print('Search Time left: ', search_time_constraint - (time.time() - timeout_start))
        graph.mark_node_visited(current_node)
        print('Current node: ', current_node.complexities)
        print('Visited')
        graph.print_visited()

        # Find current node's optimal confidence value.
        satisfy_model_time_constraint = \
            current_node.find_and_set_optimal_confidence_value_grid(\
                conf_value_inputs, conf_value_labels, model_time_constraint)

        if satisfy_model_time_constraint:
            # Get the current node's test time and test accuracy using the optimal
            # confidence value on the validation dataset. If test time satisfies
            # the input model constraint and the test accuracy is better than
            # best_accuracy_so_far, update the variables.
            test_accuracy, test_time = current_node.get_accuracy_and_time_optimal_conf(\
                    validation_inputs, validation_labels)
            
            if test_time < model_time_constraint:
                best_node_so_far = current_node
                best_accuracy_so_far = test_accuracy
                print('Best Node So Far is now: ', best_node_so_far.complexities)
                print('Test time: ', test_time, 'Test accuracy:', test_accuracy)

        # Select neighbor
        prob = np.random.random()
        if prob < epsilon:
            print('Randomly Selected Neighbor')
            current_node = graph.get_random_neighbor_pairing(current_node)
        else:
            print('Greedily Selected Neighbor')
            if satisfy_model_time_constraint:
                print('Attempting to find more complex neighbor')
                current_node = graph.get_neighbor_greater_complexity(current_node)
            else:
                print('Attempting to find simpler neighbor')
                current_node = graph.get_neighbor_simpler_complexity(current_node)

        # Traversed all nodes can traverse (may not be entire graph depending on visited set).
        if current_node == None:
            break

    return best_node_so_far, time.time() - timeout_start

def naive_search():
    pass

def main():
    print('Loading data...')
    # Get MNIST data
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    
    # For testing model to sort by complexity
    x_test_1 = x_test[:3333]
    y_test_1 = y_test[:3333]
    # For finding optimal confidence value
    x_test_2 = x_test[3334:6666]
    y_test_2 = y_test[3334:6666]
    # For validating optimal confidence value
    x_test_3 = x_test[6667:10000]
    y_test_3 = y_test[6667:10000]

    # Train each of the models 
    # input_shape = (28, 28, 1)

    # No hidden layers (logistic regression)
    # l0_model = models.get_trained_l1_all_digit_model(x_train, y_train)

    # 1 hidden layer
    # l1_model = models.get_trained_l1_all_digit_model(x_train, y_train)

    # 1 conv layer, 1 hidden layer
    # l2_model = models.get_trained_l2_all_digit_model(x_train, y_train, input_shape)

    # 2 conv layers, 1 hidden layer
    # l3_model = models.get_trained_l3_all_digit_model(x_train, y_train, input_shape)

    # Save:
    # l0_model.save('models/l0_model')
    # l1_model.save('models/l1_model')
    # l2_model.save('models/l2_model')
    # l3_model.save('models/l3_model')

    print('Loading models...')
    # l0_model = tf.keras.models.load_model('models/l0_model')
    l1_model = tf.keras.models.load_model('models/l1_model')
    l2_model = tf.keras.models.load_model('models/l2_model')
    l3_model = tf.keras.models.load_model('models/l3_model')

    # Load saved models and store as Graph.Model object (with corresponding complexity)
    print('Storing models in Graph object...')
    models = []
    for i in range(1, 4):
        current_model = tf.keras.models.load_model('models/l' + str(i) + '_model')
        current_model_object = Graph.Model(current_model)
        before_time = time.time()
        results = current_model.evaluate(x_test_1, y_test_1, verbose=0)
        after_time = time.time()

        current_model_object.accuracy = results[1]
        current_model_object.time = after_time - before_time
        current_model_object.complexity_index = i - 1

        print('complexity_index:', current_model_object.complexity_index, \
            'accuracy:', current_model_object.accuracy, 'time: ', current_model_object.time)

        models.append(current_model_object)

    # Create graph object and add models sorted based on complexity
    graph = Graph.Graph(models)

    # Call GreedyWalk
    print('Running Greedy Walk...')
    output, search_time = greedy_walk(graph, 20, 0.3, x_test_2, y_test_2, x_test_3, y_test_3)
    if output == None:
        print('Did not find viable pairing')
    else:
        print('Found', output.complexities, 'in', search_time, 'seconds')

if __name__ == '__main__':
    main()