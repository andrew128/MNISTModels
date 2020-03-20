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


# class TestGetMostSimpleNode(unittest.TestCase):
#     def test_mark_node_visited_basic(self):
#         self.assertEqual('foo'.upper(), 'FOO')

# class TestGetRandomNeighborPairing(unittest.TestCase):
#     def test_mark_node_visited_basic(self):
#         self.assertEqual('foo'.upper(), 'FOO')

# class TestGetNeighborGreaterComplexity(unittest.TestCase):
#     def test_mark_node_visited_basic(self):
#         self.assertEqual('foo'.upper(), 'FOO')

# class TestGetNeighborSimplerComplexity(unittest.TestCase):
#     def test_mark_node_visited_basic(self):
#         self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()