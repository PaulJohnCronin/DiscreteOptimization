"""Simple Travelling Salesperson Problem (TSP) between cities."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

multiplier = 1

def plot_nodes():
    global nodes
    print("PLOTTING LOOPS", datetime.now().strftime("%H:%M:%S"))
    #print("loops: ", loops)
    plt.scatter(nodes[:,0], nodes[:,1], color='hotpink')
    plt.gca().set_aspect('equal')
    plt.show()
    input("PAUSE")


def plot_loops(routing, manager, solution):
    global nodes
    print("PLOTTING LOOPS", datetime.now().strftime("%H:%M:%S"))
    #print("loops: ", loops)
    index =0
    for node in nodes:
        plt.scatter(node[0], node[1], color = 'hotpink')
        plt.annotate(index, (node[0], node[1]))
        index=index+1

    index = routing.Start(0)
    loop = []
    while not routing.IsEnd(index):
        loop.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    x_values = []
    y_values = []
    for node in loop:
        x_values.append(nodes[node, 0])
        y_values.append(nodes[node, 1])
    plt.plot(x_values, y_values, 'bo', linestyle="--")

    plt.show()
    return



def parse_nodes(input_data):
    global nodes, multiplier
    print("GETTING NODES", datetime.now().strftime("%H:%M:%S"))

    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])

    # construct a numpy 2D array of the node positions
    nodes = np.full((node_count,2),dtype=np.single, fill_value=0)

    for i in range(node_count):
        line = lines[i+1]
        parts = line.split()
        #if (float(parts[0])/div != 2*int(float(parts[0]))) or (float(parts[1])/div != 2*int(float(parts[1]))):
        #    print("FRACTION!!!", parts[0], parts[1])
        nodes[i,0]=np.single(parts[0])
        nodes[i,1]=np.single(parts[1])

    if node_count == 33810:
        multiplier = 1 # 1/16
    elif node_count == 51:
        multiplier = 10000
    else:
        multiplier = 100
    nodes = nodes * multiplier

    #plot_nodes()
    return node_count

def scipy_cdist(node_count):
    # just produces a matrix of distances between points
    global nodes
    print("COMPUTING DISTANCE MATRIX", datetime.now().strftime("%H:%M:%S"))
    if node_count == 33811: #33810
        nodes_split = np.split(nodes,30,axis=0)
        accum_matrix = cdist(nodes_split[0], nodes, metric='euclidean').astype('uint16')
        for i in range(1,30):
            temp = cdist(nodes_split[i], nodes, metric='euclidean').astype('uint16')
            accum_matrix = np.concatenate((accum_matrix, temp),axis=0)
    else:
        accum_matrix = cdist(nodes, nodes, metric='euclidean') #.astype('uint16')
    return accum_matrix

def create_data_model(dist2_matrix):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = dist2_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = ''
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {}'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    #plan_output += ' {}\n'.format(manager.IndexToNode(index))
    return plan_output

def plot_hist(dist2_matrix, node_count):
    dist3_matrix = dist2_matrix.copy()
    for i in range(node_count):
        dist3_matrix[i,i] = 65535
    print("starting histogram plot")
    temp = np.min(dist3_matrix,axis=1)
    print(temp)
    plt.hist(temp, bins =int(np.max(dist3_matrix)/1000))
    #plt.hist(nodes[:,0], bins = 10000)
    #plt.show()
    #plt.hist(nodes[:,1], bins = 10000)
    plt.show()
    print("ending histogram plot")
    input()

def find_curves(dist2_matrix, node_count):
    criteria = 3500
    for i in range(node_count):
        dist2_matrix[i,i]=65535

    dist2_criteria = dist2_matrix < criteria
    print(np.min(dist2_matrix))
    no_nodes = np.sum((dist2_criteria & 1),axis=1) ==0
    end_nodes = np.sum((dist2_criteria & 1),axis=1) ==1
    mid_nodes = np.sum((dist2_criteria & 1),axis=1) ==2
    print(len(end_nodes))
    plt.scatter(nodes[:,0], nodes[:,1], color='blue')
    plt.scatter(nodes[end_nodes,0], nodes[end_nodes,1], color='black')
    plt.scatter(nodes[end_nodes,0], nodes[end_nodes,1], color='red')
    plt.scatter(nodes[end_nodes,0], nodes[end_nodes,1], color='yellow')
    plt.show()
    quit()



def solve_it(input_data):
    # build the initial circuit
    global nodes
    node_count = parse_nodes(input_data)
    #plot_nodes()
    # plot_hist(dist2_matrix, node_count)
    #dist2_matrix = get_dist2_matrix(node_count)
    dist2_matrix = scipy_cdist(node_count)
    #find_curves(dist2_matrix,node_count)

    print("STARTING OR TOOLS")
    """Entry point of the program."""
    # Instantiate the data problem.

    data = create_data_model(dist2_matrix)
    # Create the routing index manager.

    manager = pywrapcp.RoutingIndexManager(node_count,1, 0) #number of nodes, number of vehicles, index of depot
    # Create Routing Model.

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node]+0.5)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters = routing.DefaultSearchParameters()
    #search_parameters.use_depth_first_search = True

    if node_count != 33810:
        search_parameters.local_search_operators.use_path_lns = 3
        search_parameters.local_search_operators.use_inactive_lns = 3
        #search_parameters.local_search_operators.use_light_propagation = 3
        search_parameters.local_search_operators.use_relocate = 3
        search_parameters.local_search_operators.use_relocate_pair = 3
        search_parameters.local_search_operators.use_relocate_neighbors = 3
        search_parameters.local_search_operators.use_exchange = 3
        search_parameters.local_search_operators.use_cross_exchange = 3
        search_parameters.local_search_operators.use_tsp_opt = 3
        search_parameters.local_search_operators.use_relocate_and_make_active = 3  # costly if true by default
        search_parameters.local_search_operators.use_make_chain_inactive = 3
        search_parameters.local_search_operators.use_extended_swap_active = 3
        search_parameters.local_search_operators.use_node_pair_swap_active = 3
        search_parameters.local_search_operators.use_path_lns = 3
        search_parameters.local_search_operators.use_full_path_lns = 3
        search_parameters.local_search_operators.use_tsp_lns = 3
        search_parameters.local_search_operators.use_inactive_lns = 3

        search_parameters.local_search_operators.use_global_cheapest_insertion_expensive_chain_lns = 3
        search_parameters.local_search_operators.use_local_cheapest_insertion_expensive_chain_lns = 3
        search_parameters.local_search_operators.use_global_cheapest_insertion_close_nodes_lns = 3
        search_parameters.local_search_operators.use_local_cheapest_insertion_close_nodes_lns = 3

    if node_count == 51:
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # CHRISTOFIDES 331351.73
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH) #38685.66
        search_parameters.time_limit.seconds = 50
        search_parameters.solution_limit = 50

    elif node_count == 100:
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)  # CHRISTOFIDES 331351.73
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH) #38685.66
        search_parameters.time_limit.seconds = 50
        search_parameters.solution_limit = 50

    elif node_count == 200:
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)  # CHRISTOFIDES 331351.73
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH) #38685.66
        search_parameters.time_limit.seconds = 100
        search_parameters.solution_limit = 60
        search_parameters.lns_time_limit.seconds = 200
    elif (node_count == 574) or (node_count == 1889):
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES  # CHRISTOFIDES 331351.73 FIRST_UNBOUND_MIN_VALUE
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.guided_local_search_lambda_coefficient(0.2)
#        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.use_depth_first_search
        routing_enums_pb2.LocalSearchMetaheuristic.guided_local_search_lambda_coefficient = 0.1
        #routing_enums_pb2.LocalSearchMetaheuristic.use_depth_first_search =True

        #print("use_depth_first_search", routing_enums_pb2.LocalSearchMetaheuristic.use_depth_first_search)
        #quit()

        search_parameters.time_limit.seconds = 15300
        search_parameters.solution_limit = 5000
        search_parameters.lns_time_limit.seconds = 10000

        search_parameters.local_search_operators.use_path_lns = 3
        search_parameters.local_search_operators.use_inactive_lns = 3
        #search_parameters.local_search_operators.use_light_propagation = False
        search_parameters.local_search_operators.use_relocate = 3
        search_parameters.local_search_operators.use_relocate_pair = 3
        search_parameters.local_search_operators.use_relocate_neighbors = 3
        search_parameters.local_search_operators.use_exchange =3
        search_parameters.local_search_operators.use_cross_exchange = 3
        search_parameters.local_search_operators.use_tsp_opt = 3
        search_parameters.local_search_operators.use_relocate_and_make_active = 3 # costly if true by default
        search_parameters.local_search_operators.use_make_chain_inactive = 3
        search_parameters.local_search_operators.use_extended_swap_active = 3
        search_parameters.local_search_operators.use_node_pair_swap_active = 3
        search_parameters.local_search_operators.use_path_lns = 3
        search_parameters.local_search_operators.use_full_path_lns = 3
        search_parameters.local_search_operators.use_tsp_lns = 3
        search_parameters.local_search_operators.use_inactive_lns = 3

        search_parameters.local_search_operators.use_global_cheapest_insertion_expensive_chain_lns = 3
        search_parameters.local_search_operators.use_local_cheapest_insertion_expensive_chain_lns = 3
        search_parameters.local_search_operators.use_global_cheapest_insertion_close_nodes_lns = 3
        search_parameters.local_search_operators.use_local_cheapest_insertion_close_nodes_lns = 3
    elif node_count == 33810:
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES  # CHRISTOFIDES 331351.73
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 16500
        search_parameters.solution_limit = 5000
        search_parameters.lns_time_limit.seconds = 5000
    else:
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)  # CHRISTOFIDES 331351.73
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH) #38685.66
        search_parameters.time_limit.seconds = 100
        search_parameters.solution_limit = 60
        search_parameters.lns_time_limit.seconds = 200

    search_parameters.log_search = True

    # Solve the problem.
    print("STARTING THE SOLVE")
    solution = routing.SolveWithParameters(search_parameters)

    # prepare the solution in the specified output format
    output_data = str(round(float(solution.ObjectiveValue())/multiplier,2)) + ' ' + str(0) + '\n'
    output_data += print_solution(manager, routing, solution)
    #plot_loops(routing, manager, solution)
    del dist2_matrix
    del solution
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solverOLD.py ./data/tsp_51_1)')
