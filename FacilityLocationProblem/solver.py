import math
import numpy as np
from scipy.spatial.distance import cdist
from ParseInput import get_invariants
from NQueensModel import nqueens_model
from CPMpy_nq_model import *

import PlotFacilitiesCustomers
from ortools.sat.python import cp_model
from datetime import datetime

def reduce_facilities(distance_matrix, invariants):
    facility_array = np.full(invariants.facility_count,dtype=bool, fill_value= False)
    for c in range(invariants.customer_count):
        indices = np.argsort(distance_matrix[:,c])
        for i in range(invariants.facility_reduce):
            facility_array[indices[i]]=True
    temp = np.arange(0,invariants.facility_count,dtype=int)
    print(temp[facility_array])
    print(sum(facility_array))
    return facility_array


def solve_it(input_data):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    global invariants, distance_matrix

    # get the problem invariants
    invariants = get_invariants(input_data)     # parse the input
    solution_base = np.arange(invariants.facility_count,dtype=int)

    # get the distance matrix
    print("BUILDING DISTANCE MATRIX")
    distance_matrix = cdist(invariants.facilities_location, invariants.customers_location, metric='euclidean')#.astype(np.float16)
    solution, value = cpmpy_model(distance_matrix,invariants)

    solutionX=[]
    for i in range(len(solution)):
        solutionX.append(solution[i])
        #solutionX.append(solution_base2[solution[i]])
    output_data = '%.2f' % value + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solutionX))

    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

