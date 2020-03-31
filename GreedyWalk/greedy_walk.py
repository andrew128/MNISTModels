import numpy as np
import time

import GreedyWalk.graph as Graph

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
        graph.mark_node_visited(current_node)

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

        # Select neighbor
        prob = np.random.random()
        if prob < epsilon:
            current_node = graph.get_random_neighbor_pairing(current_node)
        else:
            if satisfy_model_time_constraint:
                current_node = graph.get_neighbor_greater_complexity(current_node)
            else:
                current_node = graph.get_neighbor_smaller_complexity(current_node)

        # Traversed all nodes can traverse (may not be entire graph depending on visited set).
        if current_node == None:
            break

    return best_node_so_far

def naive_search():
    pass

def main():
    # Get MNIST data and split train-test 0.8-0.2

    # Train each of the models and store as Graph.Model object (with corresponding complexity)

    # Remove any models with higher time and worse accuracy than any other model

    # Create graph object and add models sorted based on complexity

    # Call GreedyWalk
    pass

if __name__ == '__main__':
    main()