from communities.algorithms import bron_kerbosch
import numpy as np
from FindSingleClique import find_single_clique
from FindSingleClique import convert_adj_matrix

def get_max_clique_nodes(node_count, edge_count,edges, spares):
    # build adjacent matrix
    adj_matrix = np.full((node_count,node_count),fill_value=False)
    for e in range(edge_count):
        adj_matrix[edges[e,0],edges[e,1]] = True
        adj_matrix[edges[e,1],edges[e,0]] = True
    print("STARTING BRON KERBOSH ALGORITHM")
#    communities = bron_kerbosch(adj_matrix, pivot=True)
    communities = convert_adj_matrix(node_count, adj_matrix, spares)
    print("ENDING BRON KERBOSH ALGORITHM")
    print(len(communities))
    return convert_adj_matrix(node_count, adj_matrix, spares)

