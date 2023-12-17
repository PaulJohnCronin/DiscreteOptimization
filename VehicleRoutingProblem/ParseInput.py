import numpy as np
def parse_input(input_data):
    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    customer_count = int(parts[0]) -1
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    parts = lines[1].split()
    depot = np.full((1,2),dtype=int, fill_value = 0)

    depot[0,0] = int(float(parts[1]))
    depot[0,1] = int(float(parts[2]))
    #print("DEPOT: ", depot)

    customer_positions = np.full((customer_count,2),dtype=int, fill_value = 0)
    customer_demand = np.full(customer_count,dtype=int, fill_value = 0)
    for i in range(2, customer_count +2):
        line = lines[i]
        parts = line.split()
        customer_demand[i - 2] = int(parts[0])
        customer_positions[i - 2, 0] = int(float(parts[1]))
        customer_positions[i - 2, 1] = int(float(parts[2]))
    customer_positions = customer_positions - depot
    return customer_count, vehicle_count, vehicle_capacity, customer_demand, customer_positions
