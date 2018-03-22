from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import DataHandler as dh
import random


def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


def svm_classifier(train,test):
    X = np.array(train[1:]).transpose().tolist()
    Y = train[0]

    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, Y)

    X_test = np.array(test[1:]).transpose().tolist()
    Y_test = np.array(test[0])

    output = ''
    count = 0
    for i in range(len(X_test)):
        result = clf.predict([X_test[i]])
        if result[0] == Y_test[i]:
            count += 1
        output = output + str(result[0]) + ','
    with open('SVMResult.txt', 'w') as f:
        f.write(output[:-1])
    f.close()
    return count / len(Y_test)


# This function is used to perform K-fold cross validation
# k_fold: number of splits for cross-validation, data: data for CV
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
        accuracy = svm_classifier(train, test)
        accuracy_list.append(accuracy)
        print("Iterantion", i+1, "accuracy:", accuracy)

    print("Average accuracy:", sum(accuracy_list)/len(accuracy_list))


def start(filename, class_ids, test_instances, fold):
    format_data(filename, class_ids, test_instances)

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
    '''
    # Use this code for CV using sklearn liabrary
    trainX = np.array(train[1:]).transpose()
    trainY = np.array(train[0]).transpose()
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, trainX, trainY, cv=5)
    print(scores)
    '''
    accuracy = svm_classifier(train, test)
    print("\n##################################")
    print("Overall Accuracy:", accuracy)
    return accuracy




if __name__ == "__main__":

    filename = "HandWrittenLetters.txt"

    student_id = [4, 5, 9, 1]                       # Student ID: 1001544591 ----- [4,5,9,1]
    name_id = dh.letter_2_digit_convert('NTUR')     # Nishant Thakur         ----- [N,T,U,R] -> As 'T' was repeated replace it with 'U'
    class_ids = student_id + name_id                # [4, 5, 9, 1, 14, 20, 21, 18]
    print("For class:", class_ids)
    test_instances = [0, 9]

    start(filename, class_ids, test_instances, fold = 5)