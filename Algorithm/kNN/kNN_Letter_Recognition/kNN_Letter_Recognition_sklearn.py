## kNN Letter Recognition Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/2
#
# Detail:
#   Total Data = 20000
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

def loadData(path):
    letters = pd.read_csv(path)
    data = np.array(letters.drop(['lettr'], 1))
    label = np.array(letters['lettr'])
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainKNN(data_train, label_train, k):
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(data_train, label_train)
    return kNN

def testAccuracy(data_test, label_test, kNN):
    return kNN.score(data_test, label_test)

def evaluateModel(data_test, label_test, kNN):
    print(metrics.classification_report(label_test, kNN.predict(data_test)))
    print(metrics.confusion_matrix(label_test, kNN.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/letter-recognition.csv')

    # Train Model
    kNN_model = trainKNN(data_train, label_train, 3)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, kNN_model)))

    # Evaluate Model
    evaluateModel(data_test, label_test, kNN_model)

if __name__ == '__main__':
    main()