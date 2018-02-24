import numpy as np

train = []
test = []
with open('trainDataXY.txt') as f:
    for line in f:
        data = line[:-1].split(',')
        train.append(data)

with open('testDataXY.txt') as f:
    for line in f:
        data = line[:-1].split(',')
        test.append(data)

Xtrain = train[1:]
Xtest = test[1:]
Ytrain = train[0]
Ytest = test[0]
N_train = len(Xtrain[0])
N_test = len(Xtest[0])

A_train = np.ones((1,N_train))    # N_train : number of training instance
A_test = np.ones((1,N_test))      # N_test  : number of test instance
Xtrain_padding = np.row_stack((Xtrain,A_train))
Xtest_padding = np.row_stack((Xtest,A_test))

print(Xtrain_padding)
print(Xtest_padding)

'''computing the regression coefficients'''
B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
Ytest_padding = np.dot(B_padding.T,Xtest_padding)
Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
err_test_padding = Ytest - Ytest_padding_argmax
TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
