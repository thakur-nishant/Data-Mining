import numpy as np

import DataHandler as dh

import random

def linear_regression(train, test):
    Xtrain = np.array(train[1:],dtype='float')
    Xtest = np.array(test[1:],dtype='float')
    #convert class ids to class indicator representation
    Ytrain = [[1 if j == float(train[0][i])-1 else 0 for j in range(26)] for i in range(len(train[0]))]

    Ytrain = np.array(Ytrain,dtype='float')
    Ytest = np.array(test[0], dtype='float')

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
    err_test_padding = Ytest - Ytest_padding_argmax
    TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100

    with open('LinearRegressionResult.txt', 'w') as f:
        f.write(str(Ytest_padding_argmax.tolist())[1:-1])
    f.close()

    return (TestingAccuracy_padding)


def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


def cross_validation(k_fold, data):
    data = np.array(data).transpose().tolist()
    random.shuffle(data)
    n = len(data)
    len_k = n // k_fold
    accuracy_list = []
    for i in range(k_fold):
        start = i * len_k
        end = (i + 1) * len_k
        test = data[start:end]
        train = [x for x in data if x not in test]

        train = np.array(train).transpose().tolist()
        test = np.array(test).transpose().tolist()
        accuracy = linear_regression(train, test)
        accuracy_list.append(accuracy)
        print("Fold", i+1, "accuracy:", accuracy)

    print("Average accuracy:", sum(accuracy_list)/len(accuracy_list))


def start(filename, class_ids, test_instances, fold):

    format_data(filename,class_ids, test_instances)

    train = []
    test = []
    with open('TrainingData.txt') as f:
        for line in f:
            data = line[:-1].split(',')
            train.append(data)

    with open('TestingData.txt') as f:
        for line in f:
            data = line[:-1].split(',')
            test.append(data)

    # cross_validation(fold, train)

    accuracy = linear_regression(train, test)
    print("Overall Accuracy:", accuracy)
    return accuracy



if __name__ == "__main__":
    filename = "HandWrittenLetters.txt"

    student_id = [4, 5, 9, 1]                    # Student ID: 1001544591 ----- [4,5,9,1]
    name_id = dh.letter_2_digit_convert('NTUR')  # Nishant Thakur         ----- [N,T,U,R] -> As 'T' was repeated replace it with 'U'
    class_ids = student_id + name_id             # [4, 5, 9, 1, 14, 20, 21, 18]
    print("For class:", class_ids)
    test_instances = [0, 9]

    start(filename, class_ids, test_instances, fold = 5)