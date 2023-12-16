from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist

def get_clique_sum(clique):
    global array_sum
    return sum([array_sum[int(i)] for i in clique])

def find_single_clique(node_count,graph):
    vertices = list(graph.keys())

#    best_clique_len = 0
    best_clique_sum = 0
    best_clique = []

    for rand in range(node_count):
        clique = []
        clique.append(vertices[rand])
        for v in vertices:
            if v in clique:
                continue
            isNext = True
            for u in clique:
                if u in graph[v]:
                    continue
                else:
                    isNext = False
                    break
            if isNext:
                clique.append(v)
#        if len(clique)>best_clique_len:
#            best_clique = clique
#            best_clique_len = len(clique)
#            print(rand,best_clique_len)
        if get_clique_sum(clique) > best_clique_sum:
            best_clique = clique
            best_clique_sum = get_clique_sum(clique)
            print(rand,best_clique_sum)
    return sorted(best_clique)

def convert_adj_matrix(node_count, adj_matrix, spares):
    global array_sum
    array_sum = np.sum((adj_matrix & 1), axis =0)
    #print("ARRAY SUM:", array_sum)
    graph = dict()
    for n in range(node_count):
        line = []
        for m in range(node_count):
            if adj_matrix[n,m]:
                line.append(str(m))
        graph[str(n)]=line
    clique = find_single_clique(node_count,graph)
    clique = [int(i) for i in clique]

    mask = np.full(node_count, fill_value=False)
    for c in range(node_count):
        if c in clique:
            mask[c] = True
    #print(mask)
    temp = np.sum(adj_matrix[mask,:] & 1, axis=0)
    temp2 =np.argsort(-temp)
    print("SPARES: ",spares)
    print("BEFORE SPARES: ", len(clique))

    n=0
    while spares>0:
        if temp2[n] in clique:
            n=n+1
        else:
            clique.append(temp2[n])
            spares = spares -1
    print("AFTER SPARES: ", len(clique))
    print(clique)
    return clique


