from math import *

import numpy as np

import DataHandler as dh

def centroid_classifier(train, test):
    trainX = train[1:]
    trainY = train[0]

    testX = test[1:]
    testY = test[0]

    testX = np.array(testX).transpose()

    centroid = centroid_calculate(trainX,trainY)

    count = 0
    for i in range(len(testX)):
        predict = classify(centroid[1], centroid[0], testX[i])

        if predict == testY[i]:
            count += 1

    print("Accuracy =", count/len(testX))


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
        sum += (int(x[i]) - int(y[i]))**2

    return sqrt(sum)


def format_data(filename, class_ids):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    test_instances = [30, 38]
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)



def start():
    filename = "HandWrittenLetters.txt"
    class_ids = [1, 2, 3, 4, 5]

    format_data(filename,class_ids)

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

    centroid_classifier(train, test)


if __name__ == "__main__":
    start()
