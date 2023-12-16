def bool_model(model, solver, clique, edges, node_count, edge_count, max_colors):

    print("BUILDING NODE EDGE MATRIX")
    node_edge_matrix = []
    # create a dummy matrix
    for c in range(max_colors):
        arr = []
        for n in range(node_count):
            arr.append(2)
        node_edge_matrix.append(arr)

    # add the clique values
    print("ADDING THE CLIQUE VALUES")
    clique_count =0
    for n in clique:
        for c in range(max_colors):
            node_edge_matrix[c][n] = 0
        node_edge_matrix[clique_count][n] = 1
        clique_count += 1

    # find any ones, and add zeros
    print("ONES ZEROS AND BOOLS")
    for edge in edges:
        for c in range(max_colors):
            if isinstance(node_edge_matrix[c][edge[0]], int) and (node_edge_matrix[c][edge[0]] == 1):
                node_edge_matrix[c][edge[1]] = 0
                #print("TYPE 1")
            elif isinstance(node_edge_matrix[c][edge[1]], int) and (node_edge_matrix[c][edge[1]] == 1):
                node_edge_matrix[c][edge[0]] = 0
                #print("TYPE 2")
            else:
                #print("TYPE 3")
                if isinstance(node_edge_matrix[c][edge[0]], int) and (node_edge_matrix[c][edge[0]] == 2):
                    node_edge_matrix[c][edge[0]] = model.NewBoolVar('')
                if isinstance(node_edge_matrix[c][edge[1]], int) and (node_edge_matrix[c][edge[1]] == 2):
                    node_edge_matrix[c][edge[1]] = model.NewBoolVar('')
                model.AddAtMostOne(node_edge_matrix[c][edge[0]],node_edge_matrix[c][edge[1]])

    for n in range(node_count):
        if n not in clique:
            model.AddExactlyOne([node_edge_matrix[c][n] for c in range(max_colors)])

    print("STARTING SOLVER")
    status = solver.Solve(model)
    print("ENDING SOLVER")

    solution = []
    for n in range(node_count):
        for c in range(max_colors):
            if solver.Value(node_edge_matrix[c][n]):
                solution.append(c)

    print(solution)
    return solution

