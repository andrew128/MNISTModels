# GreedyWalk

## Purpose
GreedyWalk's purpose is to find the model pairing that achieves the highest accuracy on the MNIST dataset within a certain time limit given a set of models.

## Description of Algorithm
- Inputs:
    - Search Time Constraint
    - Model Time Constraint
    - Set of Models
    - Dataset
- Output:
    - pair of models
    
## How to Run

## Directory Structure
- greedy_walk.py: implementation of GreedyWalk
- graph.py: Graph data structure that GreedyWalk uses
- graph_test.py: Tests for graph.py

## Test Cases
- TestGraphInit
    - test_graph_init_basic
    - test_graph_init_assertion_error

- TestMarkNodeVisited
    - test_mark_node_visited_basic0
    - test_mark_node_visited_basic1

- TestGetMostSimpleNode
    - test_get_most_simple_node_basic0
    - test_get_most_simple_node_basic1
    - test_get_most_simple_node_empty

- TestGetRandomNeighborPairing
    - test_get_random_neighbor_basic0
    - test_get_random_neighbor_basic1
    - test_get_random_neighbor_none

- TestHasUnvisitedGreaterComplexityNeighbors
    - test_has_unvisited_greater_complexity_neighbors0
    - test_has_unvisited_greater_complexity_neighbors1
    - test_has_unvisited_greater_complexity_neighbors_none

- TestGetNeighborGreaterComplexity
    - test_get_neighbor_greater_complexity_basic0
    - test_get_neighbor_greater_complexity_basic1
    - test_get_neighbor_greater_complexity_none0
    - test_get_neighbor_greater_complexity_none_most_complex
    - test_get_neighbor_greater_complexity_none_most_complex_0

- TestHasUnvisitedSimplerComplexityNeighbors
    - test_has_unvisited_greater_complexity_neighbors0
    - test_has_unvisited_greater_complexity_neighbors1
    - test_has_unvisited_greater_complexity_neighbors2
    - test_has_unvisited_greater_complexity_neighbors_none
    - test_has_unvisited_greater_complexity_neighbors_none0

- TestGetNeighborSimplerComplexity
    - test_get_neighbor_simpler_complexity_basic0
    - test_get_neighbor_simpler_complexity_none0
    - test_get_neighbor_simpler_complexity_none
    - test_get_neighbor_simpler_complexity_none_most_simple

## Potential Bottlenecks
- When getting more complex or simpler neighbor, code first checks to see if there actually exists a complex or simpler neighbor.
- The check is brute force, taking O(n^2) time to run where n is the number of nodes.
This is potentially a bottleneck for very large n.

- Assuming case where visited nodes forms a border between simpler and more complex nodes is improbable because search time constraint is constant. 
Possibility can still happen due to randomness.

## Future work
- Remove any models with higher time and worse accuracy than any other model   
    - Sort by complexity, loop backwards recording min time and if encounter current model with time larger than min, remove that node.