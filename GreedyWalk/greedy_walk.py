import numpy as np
import time

import GreedyWalk.graph as Graph

def get_optimal_confidence_value_grid(node, dataset, model_time_constraint, step_size = 0.1):
    '''
    For each possible confidence value between 0 and 1 with input step_size, run the 
    node with the current confidence value. Store the confidence_value that gives the best accuracy
    within the input time constraint.
    '''
    confidence_values = np.arange(0, 1.1, 0.1)
    best_confidence_value = 0
    best_accuracy = 0
    for conf_value in confidence_values:
        before = time.time()
        current_accuracy = node.accuracy(dataset, conf_value)
        after = time.time()

        if after - before < model_time_constraint:
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_confidence_value = conf_value
    
    return best_confidence_value

def get_optimal_confidence_value_random():
    pass

def greedy_walk(graph, search_time_constraint, model_time_constraint, \
                validation_dataset, conf_value_dataset, epsilon = 0.1):
    '''
    param: graph - Graph object representing all possible model pairings
    param: search_time_constraint - amount of time given to search
    param: model_time_constraint - amount of time output model pairing must 
                                    complete validation_dataset within
    param: validation_dataset - dataset given to test each model pairing
    param: conf_value_dataset - dataset given to find optimal confidence value
    output: tuple of models of size 2

    This method finds the model pairing chosen from the input model set
    that achieves the highest accuracy on the input dataset within the model 
    pairing input time constraint. The result is returned within the search input
    time constraint.
    '''
    timeout_start = time.time()

    current_node = graph.get_most_simple_pairing()
    visited_nodes = set([current_node])

    best_node_so_far = current_node
    best_accuracy_so_far = 0

    while time.time() < timeout_start + search_time_constraint:
        prob = np.random.random()
        if prob < epsilon:
            current_node = graph.get_random_neighbor_pairing(visited_nodes)
        else:
            current_optimal_conf_value = get_optimal_confidence_value_random(current_node, \
                                                    conf_value_dataset, model_time_constraint)
            current_node.optimal_confidence_value = current_optimal_conf_value

            # Test model pairing with current_optimal_conf_value
            test_accuracy, test_time = graph.test_node(current_node, current_optimal_conf_value)
            current_node.accuracy = test_accuracy
            current_node.time = test_time

            if test_time < model_time_constraint:
                if test_accuracy > best_accuracy_so_far:
                    best_node_so_far = current_node
                current_node = graph.get_neighbor_greater_complexity(visited_nodes)
            else:
                current_node = graph.get_neighbor_smaller_complexity(visited_nodes)

        visited_nodes.add(current_node)

    return best_node_so_far

def naive_search():
    pass

def main():
    # Get MNIST data and split train-test 0.7-0.3

    # Train each of the models

    # Create set of models object

    # Call GreedyWalk
    pass

if __name__ == '__main__':
    main()