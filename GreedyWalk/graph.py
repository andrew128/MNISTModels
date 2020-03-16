class Model():
    def __init__(self, model, complexity_index):
        self.model = model
        self.complexity_index = complexity_index

class Node():
    '''
    Model pairing
    '''
    def __init__(self, simple_model, complex_model):
        # Model objects
        self.simple_model = simple_model
        self.complex_model = complex_model

        self.optimal_confidence_value = None

        # Test accuracy/time using optimal confidence value
        self.accuracy = None
        self.time = None

    def accuracy(self, data, confidence_value):
        pass

class Graph():
    '''
    Graph encapsulates a set of trained models by providing each with 
    an ID starting from 0 in order of increasing complexity. Graph is
    made up of Node objects initialized at the beginning. It is not possible
    to add models to the Graph after initialization.

    Graph also offers methods to to get node pairings and neighboring node
    pairings.
    '''
    def __init__(self, models, data):
        '''
        Takes in a set of trained models and testing data.
        Stores the accuracy and time for each model on the test dataset.
        '''
        pass

    def __add_model(self):
        pass

    def get_most_simple_pairing(self):
        pass

    def get_random_neighbor_pairing(self):
        pass

    def get_neighbor_greater_complexity(self):
        pass

    def get_neighbor_smaller_complexity(self):
        pass

    def test_node(self, node, conf_value):
        pass

def main():
    pass

if __name__ == '__main__':
    main()