from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
np.set_printoptions(threshold=np.nan)

def predict(filename, k):
    train = []
    with open(filename) as f:
        for line in f:
            data = line[:-1].split(',')
            train.append(data)

    Y = np.array(train[0], dtype='int')
    X = np.array(train[1:]).transpose().tolist()
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    Kmeans_labels = kmeans.labels_
    C = confusion_matrix(Y, Kmeans_labels)
    C = C.T
    ind = linear_assignment(-C)
    C_opt = C[:, ind[:, 1]]
    acc_opt = np.trace(C_opt) / np.sum(C_opt)

    # with open('output3.txt', 'w') as f:
    #     f.write(str(C_opt) )
    # f.close()

    return C_opt,acc_opt


if __name__ == "__main__":
    filename = "HandWrittenLetters.txt"
    # filename = "ATNTFaceImages100.txt"
    # train = []
    # with open(filename) as f:
    #     for line in f:
    #         data = line[:-1].split(',')
    #         train.append(data[:100])
    # data = ""
    # for i in range(len(train)):
    #     string = ""
    #     for j in train[i]:
    #         string += str(j) + ","
    #     data += string[:-1] + "\n"
    #
    # with open('ATNTFaceImages100.txt', 'w') as f:
    #     f.write(data)
    # f.close()
    k = 26
    accuracy = predict(filename, k)
    print("Accuracy:",accuracy)

