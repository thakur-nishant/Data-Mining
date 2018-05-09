# -*- coding: utf-8 -*-
"""
Created on Mon May  7 03:28:03 2018

@author: Aniket Gade
"""

import f_test as ftest
import numpy as np
from math import *


def pickTrainingData(filename, feature_set):
    with open(filename) as f:
        data = []
        count = 0
        for line in f:
            if count == 0:
                raw_data = line[:-1].split(',')
                data.append(raw_data)
            if count-1 in feature_set:
                raw_data = line[:-1].split(',')
                data.append(raw_data)
            count+=1
        return data
    
def pickTestData(filename,feature_set):
     with open(filename) as f:
        data = []
        count = 0
    
        for line in f:
            if count in feature_set:
                raw_data = line[:-1].split(',')
                data.append(raw_data)
            count+=1
        return data
    
def kNN(k, train, test):

    trainY = train[0]
    trainX = train[1:]

    testX = np.array(test).transpose()
    output = []
    for i in range(len(testX)):
        predict = classify(trainX, trainY, testX[i], k)
        output.append(predict)
    print(output)



def classify(X, Y, x, k):
    data = np.array(X).transpose()
    distance = []
    for i in range(len(data)):
        distance.append(euclidean_distance(data[i], x))

    k_distance = sorted(range(len(distance)), key=lambda j: distance[j])[:int(k)]
    k_lable_list = [Y[i] for i in k_distance]
    majority_label = majority_element(k_lable_list)
    return majority_label


def majority_element(label_list):
    index, counter = 0, 1

    for i in range(1, len(label_list)):
        if label_list[index] == label_list[i]:
            counter += 1
        else:
            counter -= 1
            if counter == 0:
                index = i
                counter = 1

    return label_list[index]


def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (float(x[i]) - float(y[i]))**2

    return sqrt(sum)


if __name__ == '__main__':
    file_name = 'GenomeTrainXY.txt'
    raw_data = ftest.get_data(file_name)
    k = 3
    features, scores= ftest.f_test(raw_data)
    train = pickTrainingData(file_name, features)
    test = pickTestData("GenomeTestX.txt", features)
    
    kNN(k, train, test)
    
    