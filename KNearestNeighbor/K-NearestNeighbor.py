from math import *

import numpy as np

import DataHandler as dh


def start():
    dh.run("trainDataXY.txt")

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

    k = input("Enter value of K: ")

    kNN(k, train, test)


def kNN(k, train, test):

    trainY = train[0]
    trainX = train[1:]

    testY = test[0]
    testX = test[1:]

    testX = np.array(testX).transpose()

    count = 0
    for i in range(len(testX)):
        predict = classify(trainX, trainY, testX[i], k)

        if predict == testY[i]:
            count += 1

    print("Accuracy =", count/len(testX))



# X: Training data, Y: Training Class Lables, x: sample unknown data
def classify(X, Y, x, k):
    data = np.array(X).transpose()
    distance = []
    for i in range(len(data)):
        distance.append(euclidean_distance(data[i], x))

    k_distance = sorted(range(len(distance)), key=lambda j: distance[j])[:int(k)]
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
    sum = 0
    for i in range(len(x)):
        sum += (int(x[i]) - int(y[i]))**2

    return sqrt(sum)
    # return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


if __name__ == "__main__":
    start()