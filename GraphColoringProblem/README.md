My solution to the graph coloring problem, a problem of coloring connected nodes, with the goal to find the minimum number of colors, such that no connected nodes have the same color.

This problem uses Google's OR-tools, specifically their constrained programming (CP-SAT) solver.

To solve this problem, we first load the data.  Each datafile is simply a list of each connected node pair.

We create a large data matrix (edge_matrix) of connection between every node to every other node, True if connected, False if not connected.

In any CP problem, a goal is reduce the search space.  With graph theory, one can find the best "clique" for the graph.  A maximal clique is a clique that cannot be extended by including one more adjacent vertex, that is, a clique which does not exist exclusively within the vertex set of a larger clique.  This gives a great leg-up for this problem, because the solution will definitely containt the maximal clique.

Once the maximal clique is found, this reduces the search space, and we build the CP-model for Google's OR-Tools to solve.
