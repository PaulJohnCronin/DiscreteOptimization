"""Simple Travelling Salesperson Problem (TSP) between cities."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def TSP_distance(sent_dist_mat):
    data = {}
    data['distance_matrix']= sent_dist_mat.astype(int).tolist()
    #data['distance_matrix']= sent_dist_mat.tolist()
    data['num_vehicles'] = 1
    data['depot'] =0
    #print("HERE")
    #print(sent_dist_mat)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION)
    #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    solution = routing.SolveWithParameters(search_parameters)
    order =[]

    #print("SHAPE: ", np.shape(sent_dist_mat))
    index = routing.Start(0)
    plan_output = ''
    dist =0
    while not routing.IsEnd(index):
        #print(index, solution.Value(routing.NextVar(index)))
        if not routing.IsEnd(solution.Value(routing.NextVar(index))):
            dist += sent_dist_mat[index,solution.Value(routing.NextVar(index))]
        order.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    #print("PREVIOUS INDEX: ", previous_index)
    dist += sent_dist_mat[previous_index,routing.Start(0)]
    order.append(0)
    #print("ORDER: ",order)
    #print("XXX: ", dist, solution.ObjectiveValue())
    #return solution.ObjectiveValue(), order
    return dist, order
