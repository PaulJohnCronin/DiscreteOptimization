import numpy as np
from ortools.sat.python import cp_model
from ParseInput import parse_edges
from CliqueNodes import get_max_clique_nodes
import sys
np.set_printoptions(threshold=sys.maxsize)
from IntVarModel import int_var_model

def greedy_algorithm():
    return 

def solve_it(input_data):
    node_count, edge_count, edges = parse_edges(input_data)
    nodes  = [4, 50, 70, 100, 250, 500, 1000]
    #colors = [2, 6, 17, 16, 90, 16, 120]
    colors = [3, 6, 17, 16, 74, 16, 86]
    spare  = [0, 0,  0,  0,  0,  0,  0 ]
    index = nodes.index(node_count)
    clique = get_max_clique_nodes(node_count, edge_count,edges, spare[index])
    print(clique)

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    solution = int_var_model(model, solver, clique, edges, node_count, edge_count, colors[index])
    #solution = bool_model(model, solver, clique, edges, node_count, edge_count, colors[index])

    if spare[index] >0:
        # create numpy arrays of length N for each allowable color
        color_node_matrix = np.full((colors[index], node_count),fill_value=False)
        unplaced_nodes = np.full(node_count,fill_value=False)
        for n in range(node_count):
            if solution[n] < colors[index]:
                color_node_matrix[solution[n],n] = True
            else:
                unplaced_nodes[n]=True

        # place each unplaced node into the color_node_matrix with minimum or zero conflicts

        # while conflicts exist, find the most conflicted node and move it to the least conflicted color

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver_OLD.py ./data/gc_4_1)')

