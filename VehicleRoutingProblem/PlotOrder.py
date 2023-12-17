import matplotlib.pyplot as plt
import numpy as np
def plot_order(customer_positions, orders):
    cus_pos_plus = np.append(np.reshape(np.array([0,0]),(1,2)),customer_positions,axis=0)

    #plot the customers
    plt.scatter(cus_pos_plus[:,0], cus_pos_plus[:,1], marker = "H")

    for order in orders:
        x_values = []
        y_values = []
        for o in order:
            x_values.append(cus_pos_plus[o,0])
            y_values.append(cus_pos_plus[o,1])
            plt.plot(x_values, y_values, 'bo', linestyle="--")
    plt.show()
    return