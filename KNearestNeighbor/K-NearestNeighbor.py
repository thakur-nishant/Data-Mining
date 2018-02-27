from math import *

import numpy as np

import DataHandler as dh

dh.run('')

def kNN(train, test):



# X: Training data, Y: Training Class Lables, x: sample unknown data
def classify(X, Y, x, k):
    data = np.array(X).transpose()
    distance = []
    for i in range(len(data)):
        distance.append(data, x)

    k_distance = sorted(range(len(distance)), key=lambda j: distance[j])[:k]
    k_lable_list = [Y[i] for i in k_distance]
    majority_label = majority_element(k_lable_list)
    return majority_label


def majority_element(num_list):
    index, counter = 0, 1

    for i in range(1, len(num_list)):
        if num_list[index] == num_list[i]:
            counter += 1
        else:
            counter -= 1
            if counter == 0:
                index = i
                counter = 1

    return num_list[index]


def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
