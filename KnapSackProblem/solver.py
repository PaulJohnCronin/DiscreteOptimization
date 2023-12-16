#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import pandas as pd
import signal
import time
from contextlib import contextmanager
class TimeoutException(Exception): pass
import sys
sys.setrecursionlimit(20001)

class Data:
    def __init__(self, df, item_count, capacity, max_density):
        self.df = df.copy()
        self.item_count = item_count
        self.capacity = capacity
        self.best_cum_value = 0
        self.best_cum_weight = 0
        self.best_cum_value = 0
        self.best_array = np.zeros(item_count)
        self.max_density =max_density


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_df(input_data):
    # parse the input
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    df = pd.DataFrame(columns=['value', 'weight'])
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        df.loc[len(df), :] = [int(parts[0]), int(parts[1])]
    df['density'] = df['value'] / df['weight']
    # elimate rows that exceed the capacity
    #print(df.describe()) #JUST FOR NOW, WE CHECK MAX WEIGHT > CAPACITY
    return df, item_count, capacity

def optomistic_estimate(data, index, cum_value, cum_weight):
    opt_est = cum_value
    flag = False
    while (not(flag)) & (index < data.item_count):
        if (cum_weight + data.df['weight'].iloc[index]) < data.capacity:
            opt_est = opt_est + data.df['value'].iloc[index]
            cum_weight = cum_weight + data.df['weight'].iloc[index]
            index = index + 1
        elif (cum_weight + data.df['weight'].iloc[index]) >= data.capacity:
            flag = True
#            if major_printing:
#                print("opt_est over capacity:", cum_weight, cum_weight + data.df['weight'].iloc[index])
#                print("computing opt est extra bit")
            opt_est = float(opt_est) + float(data.capacity - cum_weight) * data.df['density'].iloc[index]
        else:
            flag = True
            opt_est = opt_est * data.capacity / cum_weight
    return float(opt_est)


def OE_recursive(data, index,cum_weight,cum_value, array, opt_est):

    if index == data.item_count:
        return

    if opt_est <= data.best_cum_value:
        return

    if (cum_weight + data.df['weight'].iloc[index]) <= data.capacity:
        array[index] = 1
        if (cum_value + data.df['value'].iloc[index]) > data.best_cum_value:
            data.best_cum_value = cum_value + data.df['value'].iloc[index]
            data.best_cum_weight = cum_weight + data.df['weight'].iloc[index]
            data.best_array = array.copy()
            #print(best_cum_value)
        OE_recursive(data, index+1, cum_weight+ data.df['weight'].iloc[index],
                     cum_value+ data.df['value'].iloc[index],array,opt_est)
    else:
        pass
#        if major_printing:
#            print("")
#            print("OVER CAPACITY",cum_weight + data.df['weight'].iloc[index])

    array[index] = 0
    opt_est = optomistic_estimate(data, index+1, cum_value, cum_weight)
    OE_recursive(data, index+1, cum_weight, cum_value, array, opt_est)
    return array

def weight_check(data, index,cum_weight):
    if index == data.item_count:
        return True
    if (cum_weight + data.df['weight'].iloc[data.item_count-1]) > data.capacity:
        #print("LOWEST POINT CUT-OFF")
        return True
    return False


def CS_recursive(data, index,cum_weight,cum_value, array):

    if weight_check(data, index, cum_weight):
        return

    if ((data.capacity - cum_weight) * data.max_density + cum_value) <= data.best_cum_value:
        return

    if (cum_weight + data.df['weight'].iloc[index]) <= data.capacity:
        array[index] = 1
        if (cum_value + data.df['value'].iloc[index]) > data.best_cum_value:
            data.best_cum_value = cum_value + data.df['value'].iloc[index]
            data.best_cum_weight = cum_weight + data.df['weight'].iloc[index]
            data.best_array = array.copy()

        subindex = 1
        flag = False
        while (index+subindex<data.item_count) and (cum_weight + data.df['weight'].iloc[index+subindex]) > data.capacity:
            subindex = subindex +1
        if index + subindex == data.item_count:
            return
        CS_recursive(data, index+subindex, cum_weight+
                     data.df['weight'].iloc[index], cum_value+ data.df['value'].iloc[index],array)
    else:
        pass
#        if major_printing:
#            print("")
#            print("OVER CAPACITY",cum_weight + data.df['weight'].iloc[index])

    array[index] = 0
    subindex = 1
    while (index + subindex < data.item_count) and (cum_weight + data.df['weight'].iloc[index + subindex]) > data.capacity:
        subindex = subindex + 1
    if index + subindex == data.item_count:
        return
    CS_recursive(data, index+subindex, cum_weight, cum_value, array)
    return #array

def solve_it(input_data):
    df, item_count, capacity = get_df(input_data)

    # sort dataframe by density in decending order
    max_density = max(df['density'])
    min_density = min(df['density'])

    data = Data(df,item_count, capacity, max_density)

    # this test determines which model we will use
    if ((data.max_density-min_density)/(data.max_density+min_density)) > 0.01:
        print("Optimistic Estimate model")

        # sort by highest density first
        data.df.sort_values('density', axis=0, ascending=False, inplace=True)

        # compute the optimistic estimate with relaxation
        opt_est = optomistic_estimate(data, 0, 0, 0)

        # start OE recursive search with time limit
        try:
            with time_limit(60 * 60 * 5):
                OE_recursive(data, 0, 0, 0, np.zeros(data.item_count), opt_est)
        except TimeoutException as e:
            print("Timed out!")

    else:
        # test if the data is bimodal or exponential
        print("Constrained Search Model")
        data.df.sort_values('weight', axis=0, ascending=False, inplace=True)

        # start CS recursive search with time limit
        try:
            with time_limit(60 * 60 * 5):
                for big_index in range(int(float(data.item_count)/2+1)):
                    #print("big index: ",big_index)
                    array = np.zeros(data.item_count)
                    array[big_index]=1
                    index = int(float(data.item_count)/2)
                    cum_weight = data.df['weight'].iloc[big_index]
                    cum_value = data.df['value'].iloc[big_index]
                    CS_recursive(data, index, cum_weight, cum_value, array)
                CS_recursive(data, int(float(data.item_count)/2)+2, 0, 0, np.zeros(data.item_count))
        except TimeoutException as e:
            print("Timed out!")

    data.df['results']=data.best_array.tolist()
    data.df.sort_index(inplace=True)
    data.df['results'] = data.df['results'].astype(int)

    # prepare the solution in the specified output format
    output_data = str(data.best_cum_value) + ' ' + str(0) + '\n'
    taken = data.df['results'].tolist()
    output_data += ' '.join(map(str, taken))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')