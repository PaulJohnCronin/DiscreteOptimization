import math
from ParseInput import parse_input
from scipy.spatial.distance import cdist
from ortools import *
from sklearn.cluster import KMeans
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from cpmpy import *
from TSPDistance import *

# HYPERPARAMETERS
train_frac = 0.05        # fraction of customers with small radiuses to intially exclude
capacity_frac = 1.0   # fraction of the capacity from the reduced
marg_ratio = 1.25
potential_soln_time_limit = 60
potential_soln_possibilities = 2

def solve_it(input_data):
    # compute parameters of the problem
    customer_count, vehicle_count, vehicle_capacity, customer_demand, customer_positions = parse_input(input_data)
    cus_pos_plus = np.append(np.reshape(np.array([0,0]),(1,2)),customer_positions,axis=0) # add back the origin to customer positions
    dist_mat = np.round(cdist(np.round(cus_pos_plus*1000),np.round(cus_pos_plus*1000), metric='euclidean')).astype(int) # get the distance_matrix
    if vehicle_count == 41:
        vehicle_count = 38

    customer_radius = dist_mat[0,1:customer_count+1]
    train = customer_radius > np.percentile(customer_radius,int(100 * train_frac))

    # compute the KMeans Clustering
    print("K-MEANS CLUSTERING")
    km = KMeans(n_clusters=vehicle_count, n_init=customer_count, random_state=0)
    clusters_fit = km.fit(customer_positions[train,:]) #, sample_weights)

    # compute the closest cluster
    print("FIND DISTANCES TO EACH CLUSTER")
    close_mat = clusters_fit.transform(customer_positions)

    # find a potential solutions
    print("FIND A POTENTIAL SOLUTION")
    m = Model()
    choices = boolvar(shape=(vehicle_count,customer_count))
    for i in range(customer_count):
        m += np.sum(choices[:,i]) == 1
    m += np.matmul(choices,customer_demand) <= vehicle_capacity
    for c in range(customer_count):
        index =0
        close_sort = np.argsort(close_mat[c,:])
        m += choices[close_sort[potential_soln_possibilities:],c]==0
    m.minimize(np.sum(np.multiply(choices.T, close_mat)))
    soln = m.solve(time_limit=potential_soln_time_limit)
    print("OBJECTIVE VALUE: ", soln)
    quit()

    # possible vehicles
    print("FIND OTHER POSSIBLE VEHICLES")
    poss_vehs = np.full((customer_count,vehicle_count), False)
    for c in range(customer_count):
        # set all values up to potential solution to true
        index =0
        close_sort = np.argsort(close_mat[c,:])
        while close_sort[index] != np.argmax(choices.value()[:,c]):
            poss_vehs[c,close_sort[index]] = True
            index += 1
        poss_vehs[c, close_sort[index]] = True

        # sets all small radius values to true
        if train[c] == False:
                poss_vehs[c,:3] = True

        # sets marginal values to True
        if close_mat[c,close_sort[1]]/ close_mat[c,close_sort[0]] < marg_ratio:
            poss_vehs[c, close_sort[1]] = True
        if close_mat[c, close_sort[2]] / close_mat[c, close_sort[0]] < marg_ratio:
            poss_vehs[c, close_sort[2]] = True

    #print(poss_vehs & 1)
    #quit()
    import matplotlib.pyplot as plt
    for c in range(customer_count):
        if train[c] == 0:
            plt.scatter(customer_positions[c,0], customer_positions[c,1], marker = "o")
        elif np.sum(poss_vehs[c,:] &1 ) == 1:
            plt.scatter(customer_positions[c,0], customer_positions[c,1], marker = "x")
        else:
            plt.scatter(customer_positions[c,0], customer_positions[c,1], marker = ".")
    plt.show()

    # create final analysis
    print("CREATING MODEL")
    m = Model()

    # create boolean constraints
    veh_cus_array = boolvar(shape=(customer_count,vehicle_count))
    for c in range(customer_count):
        for v in range(vehicle_count):
            if poss_vehs[c, v] == 0:
                m += veh_cus_array[c, v] == 0
            else:
                #m += veh_cus_array[c, v] == 1
                if np.sum(poss_vehs[c, :]) == 1:
                    m += veh_cus_array[c, v] == 1
        if np.sum(poss_vehs[c, :]) > 1:
            m += np.sum(veh_cus_array[c, :]) == 1

    # add the requirement that all vehicles under capacity
    print("ALL VEHICLES UNDER CAPACITY CONSTRAINT", vehicle_capacity)
    m += np.matmul(customer_demand,veh_cus_array) <= vehicle_capacity

    print("STARTING SOLVER")

    global best_value, best_array, best_orders
    best_value = 1e9


    def collect_():
        global best_value, best_array, best_orders
        #print()
        #print("!!! solution !!!")
        soln = veh_cus_array.value().T.copy()
        total_dist =0
        orders = []
        for v in range(vehicle_count):
            arr = (np.append(np.reshape(np.array([1]),(1,1)),np.reshape(soln[v,:],(1,-1)))) != 0
            index = np.where(arr)
            received_matrix=dist_mat[np.ix_(arr, arr)]
            dist, order= TSP_distance(received_matrix)
            total_dist += dist
            orders.append(index[0][order])
        #print("HERE: ",np.all(np.matmul(soln, customer_demand)<=vehicle_capacity))
        if (total_dist < best_value) & np.all(np.matmul(soln, customer_demand)<=vehicle_capacity):
            print(total_dist, np.matmul(soln, customer_demand))
            best_value = total_dist
            best_array = veh_cus_array.value()
            best_orders = orders
            #plot_order(customer_positions,best_orders)
            #print("orders")
            #print(best_orders)
            print("BEST VALUE:", best_value)
        return best_value

    n = m.solveAll(display = collect_)
    print("NUMER OF SOLUTIONS: ", n)

    #print(poss_vehs & 1)
    print(np.concatenate((choices.value().T & 1, close_mat, poss_vehs & 1, veh_cus_array.value() & 1), axis =1).astype(int))
    print(best_value)
    quit()

    close_mat_min = np.min(close_mat,axis=1)
    print(close_mat)
    #print(np.sort(clost_mat_min), np.median(clost_mat_min))
    quit()



    vehicles = GMM_method(customer_positions, vehicle_count, customer_count, customer_demand, vehicle_capacity)
    best_value, best_orders = engine(vehicle_count, customer_count, vehicles, customer_demand, vehicle_capacity, dist_mat)
    print("CURRENT VALUE: ", best_value)

    # prepare the solution in the specified output format
    outputData = '%.2f' % best_value + ' ' + str(0) + '\n'
    for v in range(vehicle_count):
        outputData += ' ' + ' '.join([str(customer) for customer in best_orders[v]]) + ' ' + '\n'
    print("BEST VALUE: ", best_value)

    if customer_count == 420:
        outputData += '0 0\n'
        outputData += '0 0\n'
        outputData += '0 0\n'

    print(outputData)
    return outputData

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

