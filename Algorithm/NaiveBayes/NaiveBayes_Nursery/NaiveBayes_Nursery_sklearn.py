## Gaussian Naive Bayes Nursery Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/9
#
# Detail:
#   Total Data = 12960
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

def loadData(path):
    inputData = pd.read_csv(path)

    # Transform 'string' into class number
    Labels = [
        ['usual', 'pretentious', 'great_pret'],
        ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
        ['complete', 'completed', 'incomplete', 'foster'],
        ['1', '2', '3', 'more'],
        ['convenient', 'less_conv', 'critical'],
        ['convenient', 'inconv'],
        ['nonprob', 'slightly_prob', 'problematic'],
        ['recommended', 'priority', 'not_recom'],
        ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
    ]

    le = LabelEncoder()

    # Somehow use np.mat to deal with shape problem down below
    dataTemp = np.mat(np.zeros((len(inputData), len(inputData.columns))))
    for colIdx in range(len(inputData.columns)):
        le.fit(Labels[colIdx])
        dataTemp[:, colIdx] = np.mat(le.transform(inputData.iloc[:, colIdx])).T

    num_data = np.array(dataTemp[:, :-1])
    num_label = np.array(dataTemp[:, -1])
    data_train, data_test, label_train, label_test = train_test_split(num_data, num_label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainDecisionTree(data_train, label_train):
    gnb = GaussianNB()
    gnb.fit(data_train, label_train)
    return gnb

def testAccuracy(data_test, label_test, gnb):
    return gnb.score(data_test, label_test)

def evaluateModel(data_test, label_test, gnb):
    print(metrics.classification_report(label_test, gnb.predict(data_test)))
    print(metrics.confusion_matrix(label_test, gnb.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/nursery.csv')

    # Train Model
    GoussianNaiveBayes = trainDecisionTree(data_train, label_train)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, GoussianNaiveBayes)))

    # Evaluate Model
    evaluateModel(data_test, label_test, GoussianNaiveBayes)

if __name__ == '__main__':
    main()