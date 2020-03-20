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
    def __init__(self, simple_model, complex_model):
        # Model objects
        self.simple_model = simple_model
        self.complex_model = complex_model

        self.complexities = (simple_model.complexity_index, complex_model.complexity_index)

        self.optimal_confidence_value = None

        # Test accuracy/time using optimal confidence value
        self.optimal_test_accuracy = None
        self.optimal_test_time = None

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
        self.model_dict = {}
        self.__add_models(models)
        
        self.num_models = len(models)
        self.full_size = self.num_models * (self.num_models - 1)

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
        return Node(model_dict[0], model_dict[1])

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
            if self.visited[(new_complexity_index_0, new_complexity_index_1)] == None \
                or self.visited[(new_complexity_index_1, new_complexity_index_0)] == None \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(model_dict[new_complexity_index_0], model_dict[new_complexity_index_1])

    def get_neighbor_greater_complexity(self, node):
        '''
        Returns Node containing random pairing of models with constraint that one of the 
        models in the Node returned is the same as one of the models in the input Node AND 
        that the model changed has a GREATER complexity than the original.

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
                # Randomly change first index to be less than original
                new_complexity_index_0 = random.randrange(complexity_index_0 + 1, self.num_models)
            else:
                # Randomly change second index to be less than original
                new_complexity_index_1 = random.randrange(complexity_index_1 + 1, self.num_models)

            # If visited or randomly generated indices are the same, continue
            if self.visited[(new_complexity_index_0, new_complexity_index_1)] == None \
                or self.visited[(new_complexity_index_1, new_complexity_index_0)] == None \
                    or new_complexity_index_0 == new_complexity_index_1:
                continue
            else:
                return Node(model_dict[new_complexity_index_0], model_dict[new_complexity_index_1])

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
                return Node(model_dict[new_complexity_index_0], model_dict[new_complexity_index_1])

def main():
    pass

if __name__ == '__main__':
    main()