from sklearn import svm
import numpy as np
import DataHandler as dh


def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


def start(filename, class_ids, test_instances):
    clf = svm.SVC()
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

    X = np.array(train[1:]).transpose().tolist()
    Y = train[0]

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, Y)


    X_test = np.array(train[1:]).transpose().tolist()
    Y_test = np.array(train[0])

    count = 0

    for i in range(len(X_test)):
        result = clf.predict([X_test[i]])
        if result[0] == Y_test[i]:
            count += 1

    print("Accuracy:", count/len(Y_test))




if __name__ == "__main__":

    # filename = input("Enter filename: ")
    filename = "HandWrittenLetters.txt"

    # class_ids = [1,2,3,4,5,6,7,8,9,10]
    # class_ids = random.sample(range(1, 27), 5)
    class_ids = [x for x in range(1, 6)]
    print("For class:", class_ids)
    test_instances = [30,38]

    start(filename, class_ids, test_instances)