from math import *

import numpy as np

import DataHandler as dh

import random


# Program initialization and split data into training and testing dataset
# We use DataHandler module to split the data as required.
def start(filename, class_ids, test_instances, fold=5):
    # print("For class id:",class_ids)
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

    k = input("Enter value of K: ")

    # cross_validation(fold, k, train)

    accuracy = kNN(k, train, test)
    print("Overall Accuracy: ", accuracy)


# This function is used for data formatting using the DataHandler
# filename: name of the file to perform data splitting, class_ids: list of selected class to test.
def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


# This function is used to perform K-fold cross validation
# k_fold: number of splits for cross-validation, k: number of neighbour to be selected, train: data for CV
def cross_validation(k_fold, k, data):
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
        accuracy = kNN(k, train, test)
        accuracy_list.append(accuracy)
        print("Iterantion", i+1, "accuracy:", accuracy)

    print("Average accuracy:", sum(accuracy_list)/len(accuracy_list))


# Used to run feed data to the classifier i.e 'classify' function and calculate the accuracy
# k: number of neighbour to be selected, train: training data, test: testing data
def kNN(k, train, test):

    trainY = train[0]
    trainX = train[1:]

    testY = test[0]
    testX = test[1:]

    testX = np.array(testX).transpose()

    output = ''
    count = 0
    for i in range(len(testX)):
        predict = classify(trainX, trainY, testX[i], k)
        if predict == testY[i]:
            count += 1

        output = output + str(predict) + ','
        with open('KNNResult.txt', 'w') as f:
            f.write(output[:-1])
        f.close()
    return count/len(testX)


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


# Function to calculate the Euclidean distance between 2 points
# x & y : vectors representing 2 points
def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (float(x[i]) - float(y[i]))**2

    return sqrt(sum)


if __name__ == "__main__":
    filename = "HandWrittenLetters.txt"
    student_id = [4, 5, 9, 1]                      # Student ID: 1001544591 ----- [4,5,9,1]
    name_id = dh.letter_2_digit_convert('NTUR')    # Nishant Thakur         ----- [N,T,U,R] -> As 'T' was repeated replace it with 'U'
    class_ids = student_id + name_id               # [4, 5, 9, 1, 14, 20, 21, 18]
    print("For class:", class_ids)
    test_instances = [0, 9]

    start(filename, class_ids, test_instances, fold=5)
