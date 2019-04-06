import sys
sys.path.append("../SVM_MNIST")
from SVM_MNIST_Multiclass_FromScratch import OneVsRestSVM
from SVM_MNIST_Binary_FromScratch import SVM

from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import numpy as np

def dataStandardlize(X):
    num_feature = np.shape(X)[1]
    for i in range(num_feature):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X

def loadData(standardlize=True):
    X, y = iris_data()

    if standardlize:
        X = dataStandardlize(X)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

def loadDataBinary(standardlize=True):
    X_temp, y_temp = iris_data()
    X = X_temp[y_temp!=2]
    y = y_temp[y_temp!=2]

    if standardlize:
        X = dataStandardlize(X)
    y[y==0] = -1

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

def trainSVM(train_X, train_y):
    clf = OneVsRestSVM(C=1, tol=0.001)
    clf.fit(train_X, train_y)
    return clf

def testAccuracy(test_X, test_y, clf):
    return clf.score(test_X, test_y)

def main():
    # Binary
    train_X, test_X, train_y, test_y = loadDataBinary()
    binarySVM = SVM()
    binarySVM.fit(train_X, train_y)
    print("Accuracy of Binary (only y==0 (as -1) & y==1) SVM is:", testAccuracy(test_X, test_y, binarySVM))

    # Sklearn
    train_X, test_X, train_y, test_y = loadData(standardlize=False)
    sklearnSVM = LinearSVC()
    sklearnSVM.fit(train_X, train_y)
    print("Accuracy of Scikit Learn Multi-class SVM is:", sklearnSVM.score(test_X, test_y))

    # Multiclass OVR
    train_X, test_X, train_y, test_y = loadData()
    OVRSVM = trainSVM(train_X, train_y)
    print("Accuracy of Multi-class SVM with OVR is:", testAccuracy(test_X, test_y, OVRSVM))
    score = []
    for _ in range(5):
        OVRSVM = trainSVM(train_X, train_y)
        score.append(OVRSVM.score(test_X, test_y))
    print("average of 5:", np.mean(score))

if __name__ == "__main__":
    main()
