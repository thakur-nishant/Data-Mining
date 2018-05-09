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
    
def centroid_classifier(train, test):
    trainX = train[1:]
    trainY = train[0]

#    testX = test[1:]
#    testY = test[0]
    output = []
    testX = np.array(test).transpose()

    centroid = centroid_calculate(trainX,trainY)
    for i in range(len(testX)):
        predict = classify(centroid[1], centroid[0], testX[i])
#        print(predict)
        output.append(predict)
    print(output)


def centroid_calculate(X, Y):
    centroid = {}
    label = []
    means = []
    # print("label:",Y)
    data = np.array(X).transpose()
    for i in range(len(Y)):
        if Y[i] in centroid:
            centroid[Y[i]].append(data[i])
        else:
            centroid[Y[i]] = [data[i]]

    for key in centroid.keys():
        key_data = centroid[key]
        label.append(key)
        means.append(calculate_average(key_data))

    return [label, means]



def calculate_average(data):
    data = np.array(data).transpose().astype(np.float)

    n = len(data[0])
    mean = []
    for row in data:
        mean.append(sum(row)/n)

    return mean


def classify(X, Y, x):
    distance = []

    for mean in X:
        distance.append(euclidean_distance(mean, x))

    nearest_centroid_index = distance.index(min(distance))

    return Y[nearest_centroid_index]


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
    
    centroid_classifier(train, test)
    
    