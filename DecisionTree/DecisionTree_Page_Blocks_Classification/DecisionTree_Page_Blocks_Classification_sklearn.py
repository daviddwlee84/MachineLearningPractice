## Decition Tree Page Blocks Classification Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/2
#
# Detail:
#   Total Data = 5473
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

from datetime import datetime # Calculate training time

def loadData(path):
    inputData = pd.read_csv(path)
    data = np.array(inputData.drop(['label'], 1))
    label = np.array(inputData['label'])
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainDecisionTree(data_train, label_train):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test)))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('DecisionTree/DecisionTree_Page_Blocks_Classification/page-blocks.csv')

    # Train Model
    startTime = datetime.now()
    DecisionTreeModel = trainDecisionTree(data_train, label_train)
    print('Training time:', str(datetime.now() - startTime))

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, DecisionTreeModel)))

    # Evaluate Model
    evaluateModel(data_test, label_test, DecisionTreeModel)

if __name__ == '__main__':
    main()