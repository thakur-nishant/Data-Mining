import numpy as np
from sklearn.decomposition import PCA

def calculate_covariance(x1,x2):
    return np.cov(x1,x2)

def calculate_mean(x):
    return np.average(x)

def calculate_eigen_value(x):
    value, vector = np.linalg.eig(x)
    return value, vector

if __name__ == "__main__":
    x1 = [8,0,10,10,2]
    x2 = [-20,-1,-19,-20,0]
    # x1 = [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
    # x2 = [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]

    print("mean(x1):",calculate_mean(x1))
    print("mean(x2):",calculate_mean(x2))

    X = calculate_covariance(x1,x2)
    print("Cov(x1,x2):\n", X)

    eigen_value, eigen_vector = calculate_eigen_value(X)

    print("eigen_values:\n", eigen_value)
    print("eigen_vectors:\n", eigen_vector)

    pca = PCA(n_components=1)
    pca.fit(X)

    print(pca.explained_variance_ratio_)
    print(pca.get_covariance())

