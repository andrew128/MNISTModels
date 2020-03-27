## How to Run

## Description of Algorithm

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
    - test_get_neighbor_greater_complexity_none

- TestGetNeighborSimplerComplexity
    - test_get_neighbor_simpler_complexity_basic0
    - test_get_neighbor_simpler_complexity_none

## Potential Bottlenecks
- When getting more complex or simpler neighbor, code first checks to see if there actually exists a complex or simpler neighbor.
- The check is brute force, taking O(n^2) time to run where n is the number of nodes.
This is potentially a bottleneck for very large n.

- Assuming case where visited nodes forms a border between simpler and more complex nodes is improbable because search time constraint is constant. 
Possibility can still happen due to randomness.