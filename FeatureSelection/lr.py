# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:59:04 2018

@author: Aniket Gade
"""

import f_test as ftest
import numpy as np


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
    
def linear_regression(train, test):
    Xtrain = np.array(train[1:],dtype='float')
    Xtest = np.array(test,dtype='float')
    #convert class ids to class indicator representation
    Ytrain = [[1 if j == float(train[0][i])-1 else 0 for j in range(4)] for i in range(len(train[0]))]

    Ytrain = np.array(Ytrain,dtype='float')
#    Ytest = np.array(test[0], dtype='float')

    N_train = len(Xtrain[0])
    N_test = len(Xtest[0])

    A_train = np.ones((1, N_train),dtype='float')    # N_train : number of training instance
    A_test = np.ones((1, N_test),dtype='float')      # N_test  : number of test instance
    Xtrain_padding = np.row_stack((Xtrain, A_train))
    Xtest_padding = np.row_stack((Xtest, A_test))


    '''computing the regression coefficients'''
    B = np.linalg.pinv(Xtrain_padding.T)
    B_padding = np.dot(B, Ytrain)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
    Ytest_padding = np.dot(B_padding.T, Xtest_padding)
    Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0)+1

#    err_test_padding = Ytest - Ytest_padding_argmax
#    TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
    print(Ytest_padding_argmax)
    return (Ytest_padding_argmax)



if __name__ == '__main__':
    file_name = 'GenomeTrainXY.txt'
    raw_data = ftest.get_data(file_name)
    features, scores= ftest.f_test(raw_data)
    train = pickTrainingData(file_name, features)
    test = pickTestData("GenomeTestX.txt", features)
    
    result = linear_regression(train, test)
#    print(result)
    
    
    