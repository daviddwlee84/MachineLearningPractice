## Student Performace Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/19
#
# Detail:
#   Total Data = 649
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

def loadData(path):
    inputData = pd.read_csv(path)

    # Data
    possibleFeature = inputData[['traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'activities', 'paid', 'internet', 'nursery', 'higher', 'romantic', 'freetime', 'goout', 'Walc', 'Dalc', 'health']]

    Labels = [
        # [str(x) for x in range(1, 5)], # Skip
        # [str(x) for x in range(1, 5)], # Skip
        # [str(x) for x in range(1, 5)], # Skip
        ['yes', 'no'],
        ['yes', 'no'],
        ['yes', 'no'],
        ['yes', 'no'],
        ['yes', 'no'],
        ['yes', 'no'],
        ['yes', 'no']
        # [str(x) for x in range(1, 6)], # Skip
        # [str(x) for x in range(1, 6)], # Skip
        # [str(x) for x in range(1, 6)], # Skip
        # [str(x) for x in range(1, 6)], # Skip
        # [str(x) for x in range(1, 6)]
    ]

    le = LabelEncoder()

    dataTemp = np.mat(np.zeros((len(possibleFeature), len(possibleFeature.columns))))
    for colIdx in range(4, 11):
        le.fit(Labels[colIdx-4])
        dataTemp[:, colIdx] = np.mat(le.transform(possibleFeature.iloc[:, colIdx])).T
        

    # Label
    label = np.zeros((len(inputData), 1))
    for i, finalGrade in enumerate(inputData.loc[:, 'G3']):
        if finalGrade >= 16:
            label[i] = 1
        elif finalGrade >= 14:
            label[i] = 2
        elif finalGrade >= 12:
            label[i] = 3
        elif finalGrade >= 10:
            label[i] = 4
        else:
            label[i] = 5

    data_train, data_test, label_train, label_test = train_test_split(dataTemp, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainAdaboost(data_train, label_train):
    clf = AdaBoostClassifier()
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test)))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/student-mat.csv')

    # Train Model
    Adaboost = trainAdaboost(data_train, label_train)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, Adaboost)))

    # Evaluate Model
    evaluateModel(data_test, label_test, Adaboost)

if __name__ == '__main__':
    main()