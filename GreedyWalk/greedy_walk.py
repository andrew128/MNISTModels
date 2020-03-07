import numpy as np
import time

import GreedyWalk.set_of_models as SetOfModels

def get_optimal_confidence_value_grid(node, dataset, model_time_constraint, step_size = 0.1):
    pass

def get_optimal_confidence_value_random():
    pass

def greedy_walk(set_of_models, search_time_constraint, model_time_constraint, \
                validation_dataset, conf_value_dataset, epsilon = 0.1):
    '''
    param: set_of_models
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

    current_node = set_of_models.get_most_simple_pairing()
    visited_nodes = set([current_node])

    best_node_so_far = current_node
    best_accuracy_so_far = 0

    while time.time() < timeout_start + search_time_constraint:
        prob = np.random.random()
        if prob < epsilon:
            current_node = set_of_models.get_random_neighbor_pairing(visited_nodes)
        else:
            current_optimal_conf_value = get_optimal_confidence_value_random(current_node, \
                                                    conf_value_dataset, model_time_constraint)
            current_node.optimal_confidence_value = current_optimal_conf_value

            # Test model pairing with current_optimal_conf_value
            test_accuracy, test_time = set_of_models.test_node(current_node, current_optimal_conf_value)
            current_node.accuracy = test_accuracy
            current_node.time = test_time

            if test_time < model_time_constraint:
                if test_accuracy > best_accuracy_so_far:
                    best_node_so_far = current_node
                current_node = set_of_models.get_neighbor_greater_complexity(visited_nodes)
            else:
                current_node = set_of_models.get_neighbor_smaller_complexity(visited_nodes)

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