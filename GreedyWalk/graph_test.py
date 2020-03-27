import unittest
from graph import Graph, Node, Model

class TestGraphInit(unittest.TestCase):
    def test_graph_init_basic(self):
        num_models = 4
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        # Check to make sure that visited only contains nodes with 
        # same model complexities
        self.assertEqual(num_models, len(graph.visited))
        for i in range(num_models):
            self.assertTrue((i, i) in graph.visited)

    def test_graph_init_assertion_error(self):
        num_models = 1
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        try:
            graph = Graph(models)
        except AssertionError:
            pass
            

class TestMarkNodeVisited(unittest.TestCase):
    def test_mark_node_visited_basic0(self):
        num_models = 4
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = Node(models[0], models[1])
        graph.mark_node_visited(node)

        self.assertTrue((0, 1) in graph.visited)
        self.assertTrue((1, 0) in graph.visited)

    def test_mark_node_visited_basic1(self):
        num_models = 12
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = Node(models[4], models[8])
        graph.mark_node_visited(node)

        self.assertTrue((8, 4) in graph.visited)
        self.assertTrue((4, 8) in graph.visited)


class TestGetMostSimpleNode(unittest.TestCase):
    def test_get_most_simple_node_basic0(self):
        num_models = 12
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = graph.get_most_simple_node()
        is_most_simple = (node.complexities[0] == 0 and node.complexities[1] == 1)\
             or (node.complexities[0] == 1 and node.complexities[1] == 0)
        self.assertTrue(is_most_simple)

    def test_get_most_simple_node_basic1(self):
        num_models = 3
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = graph.get_most_simple_node()
        is_most_simple = (node.complexities[0] == 0 and node.complexities[1] == 1)\
             or (node.complexities[0] == 1 and node.complexities[1] == 0)
        self.assertTrue(is_most_simple)

    def test_get_most_simple_node_empty(self):
        num_models = 1
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        try:
            graph = Graph(models)
        except AssertionError:
            pass

class TestGetRandomNeighborPairing(unittest.TestCase):
    def test_get_random_neighbor_basic0(self):
        '''
        Possible random neighors for 1,2:
        0, 1
        1, 0
        0, 2
        2, 0

        Run 100 times to reduce probability of randomly passing every time.
        '''
        for _ in range(10):
            num_models = 3
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[1], models[2])
            graph.mark_node_visited(node)

            random_neighbor_node = graph.get_random_neighbor_pairing(node)

            valid_set = set([(0, 1), (1, 0), (0, 2), (2, 0)])

            self.assertTrue(random_neighbor_node.complexities in valid_set)

    def test_get_random_neighbor_basic1(self):
        '''
        Possible random neighors for 3,2:
        0,2   3,0
        1,2   3,1
        4,2   3,4

        Run 1000 times to reduce probability of randomly passing every time.
        '''
        for _ in range(1000):
            num_models = 5
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[3], models[2])
            graph.mark_node_visited(node)
            random_neighbor_node = graph.get_random_neighbor_pairing(node)

            valid_set = set([(0, 2), (1, 2), (4, 2), (3, 0), (3, 1), (3, 4)])

            self.assertTrue(random_neighbor_node.complexities in valid_set)

    def test_get_random_neighbor_none(self):
        '''
        Possible random neighors for 3,2:
        0,2   3,0
        1,2   3,1
        4,2   3,4

        Run 1000 times to reduce probability of randomly passing every time.
        '''
        for _ in range(1000):
            num_models = 2
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[1], models[0])
            graph.mark_node_visited(node)
            random_neighbor_node = graph.get_random_neighbor_pairing(node)

            self.assertTrue(random_neighbor_node == None)

# Test has unvisited greater complexity nodes
class TestHasUnvisitedGreaterComplexityNeighbors(unittest.TestCase):
    def test_has_unvisited_greater_complexity_neighbors0(self):
        num_models = 5
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = Node(models[2], models[4])
        graph.mark_node_visited(node)
        self.assertTrue(graph.has_unvisited_greater_complexity_neighbors(node))

    def test_has_unvisited_greater_complexity_neighbors1(self):
        num_models = 5
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = Node(models[2], models[4])
        graph.mark_node_visited(node)
        node0 = Node(models[4], models[3])
        graph.mark_node_visited(node0)
        self.assertFalse(graph.has_unvisited_greater_complexity_neighbors(node))

    def test_has_unvisited_greater_complexity_neighbors_none(self):
        '''
        n = 5
        3, 4 is most complex node, should have any more complex neighbors
        '''
        num_models = 5
        models = []
        for i in range(num_models):
            models.append(Model(None))
            models[i].complexity_index = i

        graph = Graph(models)

        node = Node(models[4], models[3])
        graph.mark_node_visited(node)
        self.assertFalse(graph.has_unvisited_greater_complexity_neighbors(node))

class TestGetNeighborGreaterComplexity(unittest.TestCase):
    def test_get_neighbor_greater_complexity_basic0(self):
        '''
        Possible more complex neighors for 0,1:
        0, 2 and 2, 1

        Run 100 times to reduce probability of randomly passing every time.
        '''
        for _ in range(100):
            num_models = 3
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[0], models[1])
            graph.mark_node_visited(node)
            neighbor_node_greater_complexity = graph.get_neighbor_greater_complexity(node)

            valid_set = set([(0, 2), (2, 1), (2, 0), (1, 2)])

            self.assertTrue(neighbor_node_greater_complexity.complexities in valid_set)

    def test_get_neighbor_greater_complexity_basic1(self):
        '''
        n = 6
        Possible more complex neighors for 4, 3:
        5, 3 and 4, 5

        Run 100 times to reduce probability of randomly passing every time.
        '''
        for _ in range(100):
            num_models = 6
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[4], models[3])
            graph.mark_node_visited(node)
            neighbor_node_greater_complexity = graph.get_neighbor_greater_complexity(node)

            valid_set = set([(5, 3), (4, 5), (3, 5), (5, 4)])

            self.assertTrue(neighbor_node_greater_complexity.complexities in valid_set)

    def test_get_neighbor_greater_complexity_none0(self):
        '''
        Possible more complex neighors for 0,1:
        0, 2 and 2, 1

        Run 100 times to reduce probability of randomly passing every time.
        '''
        for _ in range(100):
            num_models = 3
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node0 = Node(models[2], models[0])
            node1 = Node(models[2], models[1])
            graph.mark_node_visited(node0)
            graph.mark_node_visited(node1)

            node = Node(models[0], models[1])
            graph.mark_node_visited(node)
            neighbor_node_greater_complexity = graph.get_neighbor_greater_complexity(node)

            self.assertIsNone(neighbor_node_greater_complexity)


    def test_get_neighbor_greater_complexity_none_most_complex(self):
        '''
        n = 5
        Possible more complex neighors for 4, 3: None

        Run 100 times to reduce probability of randomly passing every time.
        '''
        for _ in range(100):
            num_models = 5
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[4], models[3])
            graph.mark_node_visited(node)
            neighbor_node_greater_complexity = graph.get_neighbor_greater_complexity(node)

            self.assertIsNone(neighbor_node_greater_complexity)

    def test_get_neighbor_greater_complexity_none_most_complex_0(self):
        '''
        Possible more complex neighors for 1,2:
        0, 1
        1, 0
        0, 2
        2, 0

        Run 1000 times to reduce probability of randomly passing test every time.
        '''
        for _ in range(1000):
            num_models = 3
            models = []
            for i in range(num_models):
                models.append(Model(None))
                models[i].complexity_index = i

            graph = Graph(models)

            node = Node(models[1], models[2])
            graph.mark_node_visited(node)

            neighbor_node_greater_complexity = graph.get_neighbor_greater_complexity(node)

            self.assertIsNone(neighbor_node_greater_complexity)


# class TestGetNeighborSimplerComplexity(unittest.TestCase):
#     def test_get_neighbor_simpler_complexity_basic0(self):
#         '''
#         Possible simpler neighors for 1,2:
#         1, 0; 1, 2; 0, 2

#         Run 100 times to reduce probability of randomly passing every time.
#         '''
#         for _ in range(1000):
#             num_models = 3
#             models = []
#             for i in range(num_models):
#                 models.append(Model(None))
#                 models[i].complexity_index = i

#             graph = Graph(models)

#             node = Node(models[1], models[2])
#             graph.mark_node_visited(node)
#             random_neighbor_node = graph.get_neighbor_greater_complexity(node)

#             valid_set = set([(1, 0), (2, 1), (0, 2), (2, 0), (1, 2), (0, 1)])

#             self.assertTrue(random_neighbor_node.complexities in valid_set)

#     def test_get_neighbor_simpler_complexity_none(self):
#         '''
#         Possible random neighors for 1,2:
#         0, 1
#         1, 0
#         0, 2
#         2, 0

#         Run 1000 times to reduce probability of randomly passing test every time.
#         '''
#         for _ in range(1000):
#             num_models = 3
#             models = []
#             for i in range(num_models):
#                 models.append(Model(None))
#                 models[i].complexity_index = i

#             graph = Graph(models)

#             node = Node(models[0], models[1])
#             graph.mark_node_visited(node)
#             random_neighbor_node = graph.get_neighbor_greater_complexity(node)

#             self.assertIsNone(random_neighbor_node)

if __name__ == '__main__':
    unittest.main()