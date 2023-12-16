This code solves the famous knapsack problem - please see the PDF file handout.PDF for an explanation of this problem.

When this code is called, you pass it a datafile, which is structured to contain a number of items value and weight.

A knapsack can handle a certain amount of weight.  Our goal is to maximize the value of the elements placed in the knapsack.

We first load the data, and for each item, compute it's value density, that is, it's value divided by it's weight.

There are two different models used to solve this problem.  They are:
* An optimistic estimate solution (where the difference between the max and min value density is large), or
* A constrained search solution.

Both solution mechanisms will work, but this choice just finds the correct answer faster.

Obviously, the optimistic estimate solution is a recursive method down all possible paths in a width-first search.  If a particular branch, given the best optomistic estimate of that branch, can not exceed the current best solution, that entire branch can be dropped from the search pattern.  This search space branch pruning is very efficient, resulting in a quick solution.

The constrained search solution is also a recursive search mechanism, but with a time check, to ensure that some nightmare data does not exceed possible expectations.

This code was able to find the best solution for all the different sets of data in a few moments.

