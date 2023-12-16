import numpy as np
def parse_edges(input_data):
    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    node_edge_count = np.full((edge_count,2), fill_value=0)
    # construct a numpy 2D array of the edges
    edges = np.full((edge_count,2), fill_value=0)
    for i in range(edge_count):
        line = lines[i+1]
        parts = line.split()
        edges[i,0] = int(parts[0])
        edges[i,1] = int(parts[1])
    return node_count, edge_count, edges
