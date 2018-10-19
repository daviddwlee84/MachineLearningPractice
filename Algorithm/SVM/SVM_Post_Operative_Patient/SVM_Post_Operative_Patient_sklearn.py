## SVM Post Operative Patient Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/19
#
# Detail:
#   Total Data = 90
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

def loadData(path):
    inputData = pd.read_csv(path)

    inputData = inputData.fillna(round(inputData.mean()))
    
    # Transform 'string' into class number
    Labels = [
        ['high', 'mid', 'low'],
        ['high', 'mid', 'low'],
        ['excellent', 'good', 'fair', 'poor'],
        ['high', 'mid', 'low'],
        ['stable', 'mod-stable', 'unstable'],
        ['stable', 'mod-stable', 'unstable'],
        ['stable', 'mod-stable', 'unstable'],
        [str(x) for x in range(21)], # Skip
        ['I', 'S', 'A']
    ]

    le = LabelEncoder()

    dataTemp = np.mat(np.zeros((len(inputData), len(inputData.columns))))
    for colIdx in range(len(inputData.columns)):
        if colIdx != 7:
            le.fit(Labels[colIdx])
            dataTemp[:, colIdx] = np.mat(le.transform(inputData.iloc[:, colIdx])).T
        else:
            # Skip number attribute
            dataTemp[:, colIdx] = np.mat(inputData.iloc[:, colIdx]).T

    num_data = np.array(dataTemp[:, :-1])
    num_label = np.array(dataTemp[:, -1])
    data_train, data_test, label_train, label_test = train_test_split(num_data, num_label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainSVM(data_train, label_train):
    clf = SVC(gamma='auto')
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test)))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/post-operative.csv')

    # Train Model
    SVM = trainSVM(data_train, label_train)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, SVM)))

    # Evaluate Model
    evaluateModel(data_test, label_test, SVM)

if __name__ == '__main__':
    main()