import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment


# Function to calculate the Euclidean distance between 2 points
# x & y : vectors representing 2 points
def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (float(x[i]) - float(y[i]))**2

    return np.sqrt(sum)


def nearest_centroid(centroids, x):
    distance = [euclidean_distance(mean, x) for mean in centroids]
    nearest_centroid_index = distance.index(min(distance))

    return nearest_centroid_index


def form_cluster(X, k):
    cluster_centroid = random.sample(X,k)
    while True:
        Y = []
        prev_cluster_centroid = cluster_centroid
        new_cluster = [[] for i in range(k)]
        for point in X:
            nearest_centroid_index = nearest_centroid(cluster_centroid, point)
            new_cluster[nearest_centroid_index].append(point)
            Y.append(nearest_centroid_index)

        for i,cluster in enumerate(new_cluster):
            sum = np.array([0 for j in cluster[0]],dtype='float')
            for point in cluster:
                sum += np.array(point, dtype='float')

            cluster_centroid[i] = sum/len(cluster)

        if cluster_centroid == prev_cluster_centroid:
            break

    return Y


def start(filename, k):
    data = []

    with open(filename) as f:
        for line in f:
            row = line[:-1].split(',')
            data.append(row)

    Y = np.array(data[0], dtype='int')
    X = np.array(data[1:]).transpose().tolist()

    predicted_labels = form_cluster(X, k)

    C = confusion_matrix(Y, predicted_labels)
    C = C.T
    ind = linear_assignment(-C)
    C_opt = C[:, ind[:, 1]]
    acc_opt = np.trace(C_opt) / np.sum(C_opt)
    print(acc_opt)


if __name__ == '__main__':
    # filename = 'ATNTFaceImages400.txt'
    filename = "ATNTFaceImages100.txt"
    k = 10
    start(filename, k)
