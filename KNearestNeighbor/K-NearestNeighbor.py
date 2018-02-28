from math import *

import numpy as np

import DataHandler as dh


# Program initialization and split data into training and testing dataset
# We use DataHandler module to split the data as required.
def start():
    # dh.run("trainDataXY.txt")
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

    k = input("Enter value of K: ")

    kNN(k, train, test)


def format_data(filename, class_ids):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    test_instances = [30, 38]
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


# Used to run feed data to the classifier i.e 'classify' function and calculate the accuracy
# k: number of neighbour to be selected, train: training data, test: testing data
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


# Used to classify the given unknown data point to a label/class
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


# Used to select the label/class that has majority in the list
# label_list: List of k selected nearest neighbour
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
        sum += (int(x[i]) - int(y[i]))**2

    return sqrt(sum)


if __name__ == "__main__":
    start()