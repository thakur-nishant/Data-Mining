# -*- coding: utf-8 -*-
"""
Created on Mon May  7 03:28:03 2018

@author: Aniket Gade
"""

import f_test as ftest
import numpy as np
from sklearn import svm
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
    
def svm_classifier(train,test):
    X = np.array(train[1:]).transpose()
    Y = train[0]

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, Y)
    output = []
    X_test = np.array(test).transpose()
#    print(len(X_test))
    for i in range(len(X_test)):
        result = clf.predict([X_test[i]])
        print(result)
#        output.append(result)
#    print(output)


if __name__ == '__main__':
    file_name = 'GenomeTrainXY.txt'
    raw_data = ftest.get_data(file_name)
    k = 3
    features, scores= ftest.f_test(raw_data)
    train = pickTrainingData(file_name, features)
    test = pickTestData("GenomeTestX.txt", features)
    
    svm_classifier(train, test)
    
    