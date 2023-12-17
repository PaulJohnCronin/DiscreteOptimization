import numpy as np
#from cpmpy import *
from ortools.sat.python import cp_model

import sys
np.set_printoptions(threshold=sys.maxsize)

def cpmpy_model(distance_matrix, invariants):
    # Construct the model.
    print("BUILDING MODEL")

    # select cut-off for facilities
    #distance_matrix=distance_matrix.astype(int)
    min_dist = np.percentile(distance_matrix,invariants.percentile,axis=0) # this reduces the number of available facilities

    fac_num = invariants.facility_count
    cus_num = invariants.customer_count
    fac_cap = invariants.facilities_capacity
    cus_dem = invariants.customers_demand
    fac_set = invariants.facilities_setup_cost
    run_time = invariants.run_time

    print("RUN TIME:", run_time)
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60*60*run_time #sets the maximum run time (doesn't include pre-processing)

    # build decision variables (fac_cus_matrix & fac_open_array)
    print("BUILDING INTERMEDIATE AND DECISION VARIABLES")
    fac_cus_matrix = []
    fac_open_array = []

    # build the facility / customers binary matrix with minimal variables
    for f in range(fac_num):
        line =[]
        for c in range(cus_num):
            if distance_matrix[f,c] < min_dist[c]:
                line.append(model.NewBoolVar(""))
            else:
                line.append(0)
        fac_cus_matrix.append(line)

    # build the facility possible and facility open arrays with minimal variables
    fac_possible = []
    for f in range(fac_num):
        if (sum([isinstance(fac_cus_matrix[f][c], int) for c in range(cus_num)])) == cus_num:
            fac_possible.append(False)
            fac_open_array.append(0)
        else:
            fac_possible.append(True)
            fac_open_array.append(model.NewBoolVar(""))
            model.AddMaxEquality(fac_open_array[f], fac_cus_matrix[f])
            model.Add(sum([fac_cus_matrix[f][c] * cus_dem[c] for c in range(cus_num)]) <= fac_cap[f])
    #print(fac_cus_matrix)
    #print(fac_open_array)

    # add constraint that each customer must have exactly one facility
    for c in range(cus_num):
        model.AddExactlyOne([fac_cus_matrix[f][c] for f in range(fac_num)])
        #model.Add(sum([fac_cus_matrix[f][c] for f in range(fac_num)])==1)

    # build out intermediate variable
    print("ADDING INTERMEDIATE CONSTRAINTS")

    # add constraints that facilites must stay below their capacity
    for f in range(fac_num):
        if fac_possible[f]:
            model.Add(sum([fac_cus_matrix[f][c]*cus_dem[c] for c in range(cus_num)])<=fac_cap[f])

    #compute the distance and facility set-up costs
    cost_dist = sum([fac_cus_matrix[f][c] * distance_matrix[f, c] for f in range(fac_num) for c in range(cus_num)])
    cost_setup =sum([fac_open_array[f]*fac_set[f] for f in range(fac_num)])

    model.Minimize(cost_dist+cost_setup)

    #print('NUMBER OF CONSTRAINTS: ', solver.Constraints())
    print("STARTING SOLVER")
    status=solver.Solve(model)
    print("FINISHED SOLVER")

    res = []
    for c in range(cus_num):
        list =[]
        for f in range(fac_num):
            list.append(solver.Value(fac_cus_matrix[f][c]))
        res.append(list.index(1))

    pos = np.full(cus_num,dtype=int,fill_value=0)
    for c in range(cus_num):
        search_dist = distance_matrix[res[c],c]
        sorted = np.sort(distance_matrix[:,c])
        i, = np.where(np.isclose(sorted, search_dist))  # floating-point
        pos[c] = i[0]
    print(pos)
    print("MINIMUM VALUES: ", np.max(pos))

    return res, solver.ObjectiveValue()
