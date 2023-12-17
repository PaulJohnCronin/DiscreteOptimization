import matplotlib.pyplot as plt

def plot_facilities_customers(facilities_location, customers_location, customer_array):
    plt.scatter(facilities_location[:,0], facilities_location[:,1], marker = "H")
    #print(facilities_location)
    #plt.scatter(nodes[:,0], nodes[:,1], color='hotpink')

    plt.scatter(customers_location[:,0], customers_location[:,1], marker = "o")
    #print(customers_location)

    index = 0
    for customer in customer_array:
        x_values = [facilities_location[int(customer_array[index]),0], customers_location[index,0]]
        y_values = [facilities_location[int(customer_array[index]),1], customers_location[index,1]]
        index +=1
        plt.plot(x_values, y_values, 'bo', linestyle="--")

    plt.gca().set_aspect('equal')
    plt.show()
    input("PRESS ANY KEY TO CONTINUE")
    return