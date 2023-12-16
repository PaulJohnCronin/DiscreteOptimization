from ortools.linear_solver import pywraplp
import numpy as np
from ortools.sat.python import cp_model

def int_var_model(model, solver, clique, edges, node_count, edge_count, max_colors):
    # edge constraints -> matrix
    print("CREATING EDGE CONSTRAINT MATRIX")
    edge_matrix = np.full((node_count,node_count), fill_value=False)
    for e in range(edge_count):
        edge_matrix[edges[e,0], edges[e,1]] = True
        edge_matrix[edges[e,1], edges[e,0]] = True
    print(edge_matrix&1)

    # set the clique values
    print("SETTING THE CLIQUE VALUES")
    node_color = []
    clique_color = 0
#    for n in range(node_count):
#        if n in clique:
#            node_color.append(clique_color)
#            clique_color += 1
#        else:
#            node_color.append(-1)  # the number -1 here is just a dummy variable to indicate a placeholder
#    print(node_color)

    # create the reduced domains
    print("CREATE REDUCED DOMAINS")
    for n in range(node_count):
        domain = list(range(max_colors))
        print("n: ",n)
        for m in range(n):
            if (m in clique) and edge_matrix[n,m]:
                #print("REMOVING: ", m)
                domain.remove(node_color[m])
        if n in clique:
            node_color.append(clique_color)
            clique_color += 1
        else:
            node_color.append(model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain),''))
    print(node_color)

    # create edge constraints
    for n in range(edge_count):
        if (int(edges[n, 0]) in clique) and (int(edges[n, 1]) in clique):
            pass
        else:
            model.Add(node_color[edges[n, 0]] != node_color[edges[n, 1]])

    print("STARTING THE SOLVER")

    #solver.parameters.max_time_in_seconds = 60 * 60 *2
    status = solver.Solve(model)
    print(status)
    print("ENDING THE SOLVER")
    #print(solver.Value(W))
    # build the output
    solution = []
    #solver.Value(node_color)
    for n in range(node_count):
        solution.append(solver.Value(node_color[n]))
    return solution