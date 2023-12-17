import numpy as np
import Parameters
class Invariants():
    def __init__(self, facilities_setup_cost, facilities_capacity, facilities_location,
                 customers_demand, customers_location, facility_count, customer_count,
                 total_customer_demand):
        self.facilities_setup_cost = facilities_setup_cost
        self.facilities_capacity = facilities_capacity
        self.facilities_location = facilities_location
        self.customers_demand = customers_demand
        self.customers_location = customers_location
        self.facility_count = facility_count
        self.customer_count = customer_count
        self.total_customer_demand = total_customer_demand
        params = Parameters.parameters[str(total_customer_demand)]
#        self.excess_capacity = params[1] * total_customer_demand
        self.percentile = params[1]
        self.run_time = params[2]
        self.STOP = params[3]
        self.facility_reduce = params[4]
        self.search_order = np.flip(np.argsort(facilities_capacity, axis=0))  # this is the order of searching

def get_invariants(input_data):
    print("GETTING INVARIANTS")
    #global facilities_setup_cost, facilities_capacity, facilities_location
    #global customers_demand, customers_location
    #global facility_count, customer_count, total_customer_demand
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    # establish facility information
    facilities_setup_cost = np.zeros(facility_count, dtype = float) #np.full((facility_count,1),dtype=float, fill_value=0)
    facilities_capacity   = np.zeros(facility_count, dtype = int) #np.full((facility_count,1),dtype=int, fill_value=0)
    facilities_location   = np.full((facility_count,2),dtype=float, fill_value=0)
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        #print(parts)
        facilities_setup_cost[i-1]  = float(parts[0])
        facilities_capacity[i - 1]  = int(  parts[1])
        facilities_location[i - 1, 0] = float(parts[2])
        facilities_location[i - 1, 1] = float(parts[3])
    #print("number of facilities: ", facilities_capacity)
    #print("number of customers: ", customer_count)

    # establish customer information
    total_customer_demand = int(0)
    customers_demand = np.zeros(customer_count, dtype = int) #np.full((customer_count,1),dtype=int, fill_value=0)
    customers_location   = np.full((customer_count,2),dtype=float, fill_value=0)
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        total_customer_demand += int(parts[0])
        customers_demand[i-1-facility_count]  = int(parts[0])
        customers_location[i - 1-facility_count, 0] = float(parts[1])
        customers_location[i - 1-facility_count, 1] = float(parts[2])
    print("TOTAL CUSTOMER DEMAND: ", total_customer_demand)

    return Invariants(facilities_setup_cost, facilities_capacity, facilities_location,
                      customers_demand, customers_location, facility_count, customer_count,
                      total_customer_demand)

