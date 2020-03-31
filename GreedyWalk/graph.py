import random
import time
import numpy as np

class Model():
    def __init__(self, model):
        self.model = model
        self.complexity_index = None
        self.accuracy = None
        self.time = None

class Node():
    '''
    Model pairing
    '''
    def __init__(self, model_1, model_2):
        # Model objects
        self.model_1 = model_1
        self.model_2 = model_2

        self.complexities = (model_1.complexity_index, model_2.complexity_index)

        self.optimal_confidence_value = None

    def find_and_set_optimal_confidence_value_grid(self, dataset, model_time_constraint, \
            step_size = 0.1):
        '''
        For each possible confidence value between 0 and 1 with input step_size, run the 
        node with the current confidence value. Store the confidence_value that gives the best accuracy.

        :return: boolean - true if node using optimal confidence value ran within the model's
                           time constraint.
        '''
        confidence_values = np.arange(0, 1.1, 0.1)
        best_confidence_value = 0
        best_accuracy = 0

        satisfy_model_time_constraint = False
        for conf_value in confidence_values:
            before = time.time()
            current_accuracy = self.accuracy(dataset, conf_value)
            after = time.time()

            if after - before < model_time_constraint:
                satisfy_model_time_constraint = True
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_confidence_value = conf_value

        # Only set optimal fields if within input time constraint.
        if satisfy_model_time_constraint:
            self.optimal_confidence_value = best_confidence_value
        
        return satisfy_model_time_constraint

    def find_and_set_optimal_confidence_value_random(self, dataset, model_time_constraint, step_size = 0.1):
        pass

    def accuracy(self, dataset, conf_value):
        '''
        Get the accuracy of this combined model with the given confidence value.
        '''
        pass

    def get_accuracy_and_time_optimal_conf(self, dataset):
        before = time.time()
        accuracy = self.accuracy(dataset, self.optimal_confidence_value)
        after = time.time()

        return (accuracy, after - before)

class Graph():
    '''
    Graph encapsulates a set of trained models by providing each with 
    an ID starting from 0 in order of increasing complexity. Graph is
    made up of Node objects initialized at the beginning. 

    Graph also offers methods to to get node pairings and neighboring node
    pairings.

    NOTE: It is not possible to add models to the Graph after initialization.
    '''
    def __init__(self, models):
        '''
        Takes in a set of trained models and testing data.
        '''
        # Dictionary of complexity (int) -> Model (object)
        assert len(models) >= 2

        self.model_dict = {}
        self.__add_models(models)
        
        self.num_models = len(models)
        # Counts nodes with same simple and complex complexities because
        # will exist in visited set (basically everything in 
        # num_models x num_models matrix).
        self.full_size = self.num_models * self.num_models

        # Dictionary of visited mapping [simple complexity index, complex complexity index] -> Node
        self.visited = {}

        # Add Nodes with same indices to visited so never consider them.
        for i in range(self.num_models):
            self.visited[(i, i)] = None


    def __add_models(self, models):
        for model in models:
            assert model.complexity_index != None
            self.model_dict[model.complexity_index] = model

    def mark_node_visited(self, node):
        complexities = node.complexities
        self.visited[(complexities[0], complexities[1])] = node
        self.visited[(complexities[1], complexities[0])] = node

    def get_most_simple_node(self):
        if self.num_models < 2:
            raise Exception("Error getting most simple pairing - Graph contains less than 2 models")
        return Node(self.model_dict[0], self.model_dict[1])

    def get_random_neighbor_pairing(self, node):
        '''
        Returns Node containing random pairing of models with constraint that one of the 
        models in the Node returned is the same as one of the models in the input Node.
        Continually selects random models until ends up with pair with non equal complexity
        indices and not seen before in visited.
        '''
        complexity_index_0 = node.complexities[0]
        complexity_index_1 = node.complexities[1]

        while True:
            if len(self.visited) == self.full_size:
                return None

            change_first_index = random.randrange(0, 2)
            new_complexity_index_0 = complexity_index_0
            new_complexity_index_1 = complexity_index_1

            if change_first_index:
                # Randomly change first index
                new_complexity_index_0 = random.randrange(0, self.num_models)
            else:
                # Randomly change second index
                new_complexity_index_1 = random.randrange(0, self.num_models)

            # If visited or randomly generated indices are the same, continue
            if (new_complexity_index_0, new_complexity_index_1) in self.visited \
                or (new_complexity_index_1, new_complexity_index_0) in self.visited \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(self.model_dict[new_complexity_index_0], self.model_dict[new_complexity_index_1])

    def has_unvisited_greater_complexity_neighbors(self, node):
        '''
        Returns true if there are nodes more complex than the input nodes
        that are unvisited.
        '''
        for i in range(node.complexities[0], self.num_models):
            for j in range(node.complexities[1], self.num_models):
                if (i, j) not in self.visited or (j, i) not in self.visited:
                    return True

        return False

    def get_neighbor_greater_complexity(self, node):
        '''
        Returns Node containing random pairing of models with constraint that one of the 
        models in the Node returned is the same as one of the models in the input Node AND 
        that the model changed has a GREATER complexity than the original.

        Continually selects random models until ends up with pair with non equal complexity
        indices and not seen before in visited.

        NOTE: Method doesn't automatically add returned node to visited set.
        '''
        # Check to see if actually possible to return a neighbor of greater complexity
        if len(self.visited) == self.full_size or \
                not self.has_unvisited_greater_complexity_neighbors(node):
            return None

        complexity_index_0 = node.complexities[0]
        complexity_index_1 = node.complexities[1]

        # Randomly choose index of Node's models going to change 
        while True:
            change_first_index = random.randrange(0, 2)
            new_complexity_index_0 = complexity_index_0
            new_complexity_index_1 = complexity_index_1

            # If selected to change first index and can change
            if change_first_index and complexity_index_0 + 1 < self.num_models:
                # Randomly change first index to be greater than original
                new_complexity_index_0 = random.randrange(complexity_index_0 + 1, self.num_models)
            elif not change_first_index and complexity_index_1 + 1 < self.num_models:
                # Randomly change second index to be greater than original
                new_complexity_index_1 = random.randrange(complexity_index_1 + 1, self.num_models)

            # If visited or randomly generated indices are the same, continue
            if (new_complexity_index_0, new_complexity_index_1) in self.visited \
                or (new_complexity_index_1, new_complexity_index_0) in self.visited \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(self.model_dict[new_complexity_index_0], self.model_dict[new_complexity_index_1])

    def has_unvisited_simpler_complexity_neighbors(self, node):
        '''
        Returns true if there are nodes more complex than the input nodes
        that are unvisited.
        '''
        for i in range(0, node.complexities[0] + 1):
            for j in range(0, node.complexities[1] + 1):
                if (i, j) not in self.visited or (j, i) not in self.visited:
                    return True

        return False

    def get_neighbor_simpler_complexity(self, node):
        '''
        Returns Node containing random pairing of models with constraint that one of the 
        models in the Node returned is the same as one of the models in the input Node AND 
        that the model changed has a SMALLER complexity than the original.

        Continually selects random models until ends up with pair with non equal complexity
        indices and not seen before in visited.
        '''
        # Check to see if actually possible to return a neighbor of smaller complexity
        if len(self.visited) == self.full_size or \
                not self.has_unvisited_simpler_complexity_neighbors(node):
            return None

        complexity_index_0 = node.complexities[0]
        complexity_index_1 = node.complexities[1]

        # Randomly choose index of Node's models going to change 
        while True:
            change_first_index = random.randrange(0, 2)
            new_complexity_index_0 = complexity_index_0
            new_complexity_index_1 = complexity_index_1

            # If selected to change first index and can change
            if change_first_index and complexity_index_0 >= 0:
                # Randomly change first index to be greater than original
                new_complexity_index_0 = random.randrange(0, complexity_index_0)
            elif not change_first_index and complexity_index_1 >= 0:
                # Randomly change second index to be greater than original
                new_complexity_index_1 = random.randrange(0, complexity_index_1)

            # If visited or randomly generated indices are the same, continue
            if (new_complexity_index_0, new_complexity_index_1) in self.visited \
                or (new_complexity_index_1, new_complexity_index_0) in self.visited \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(self.model_dict[new_complexity_index_0], self.model_dict[new_complexity_index_1])

def main():
    pass

if __name__ == '__main__':
    main()