# -*- coding: utf-8 -*-

import f_test as ftest
import knn as knn
import centroid as cc
import lr as lr
import svm as svm


if __name__ == '__main__':
    file_name = 'GenomeTrainXY.txt'
    raw_data = ftest.get_data(file_name)
    features, scores = ftest.f_test(raw_data)
    ftest.print_scores(features, scores)
    train = knn.pickTrainingData(file_name, features)
    test = knn.pickTestData("GenomeTestX.txt", features)
    print("\n\nPredictions for KNN (k=3) Classifier: ")
    knn.kNN(3, train, test)
    print("\nPredictions for Centroid Classifier: ")
    cc.centroid_classifier(train, test)
    print("\nPredictions for Linear Regression: ")
    lr.linear_regression(train, test)
    print("\nPredictions for SVM: ")
    svm.svm_classifier(train, test)
    