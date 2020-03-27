import random

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

        # Test accuracy/time using optimal confidence value
        self.optimal_test_accuracy = None
        self.optimal_test_time = None

    def accuracy(self, dataset, conf_value):
        pass

    def set_optimal_time_and_accuracy(self, data):
        '''
        Sets self.optimal_test_accuracy and self.optimal_test_time values
        using self.optimal_confidence_value.

        :return: void
        '''
        assert self.optimal_confidence_value != None

        pass

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

    def get_neighbor_smaller_complexity(self, node):
        '''
        Returns Node containing random pairing of models with constraint that one of the 
        models in the Node returned is the same as one of the models in the input Node AND 
        that the model changed has a SMALLER complexity than the original.

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

            # Handle case where input node is most complex viable node.
            # if complexity_index_0 == 0 or \
            #         complexity_index_1 + 1 >= self.num_models:
            #     return None

            if change_first_index:
                # Randomly change first index to be less than original
                new_complexity_index_0 = random.randrange(0, complexity_index_0)
            else:
                # Randomly change second index to be less than original
                new_complexity_index_1 = random.randrange(0, complexity_index_1)

            # If visited or randomly generated indices are the same, continue
            if self.visited[(new_complexity_index_0, new_complexity_index_1)] == None \
                or self.visited[(new_complexity_index_1, new_complexity_index_0)] == None \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(self.model_dict[new_complexity_index_0], self.model_dict[new_complexity_index_1])

def main():
    pass

if __name__ == '__main__':
    main()