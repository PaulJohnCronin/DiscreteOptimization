My solution to the facility location problem.

All the datafiles contain formatted data listing the set-up cost of a processing facility, it's capacity contraint, and its x,y location, as well as the customer demand information, including their demand and x,y location.

The goal is to use a fleet of vehicles for each facility to collect from each customer.  You may choose to open, or not open a facility, and which facilility will service which customers.

This problem was also solved using the Google, OR-Tools CP-model.  The challenge was building the model, such that it could be solved in a reasonable time.

First, we build a distance matrix from each possible facility to each customer.
Next, we reduce the number of possible facilities by setting a minimum distance to each customer - need to be careful with this as some datasets are tricky.
We then start to build the CP model.  This involves two data structures: a Boolean array of facilities if they are open or not, and a Boolean matrix, assigning each customer to a facility.
To solve the problem in a realistic time, a great number of logical constraints can be added to the model, such as 
* a customer can be serviced only by one facility, and
* that facilities must stay below their capacity limit.

Finally, we set the model to minimize the total cost of servicing the customers and the cost of setting up each facility. 
