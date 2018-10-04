## kNN Letter Recognition From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/2
#
# Detail:
#   Total Data = 20000
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from collections import Counter # Find 

from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

class kNNClassifier:
    # Initialize with k
    def __init__(self, k):
        self.__k = k
    
    # Just input the training dataset
    # Actually, the training dataset is the model itself
    def fit(self, training_data, training_label):
        self.__training_data = training_data
        self.__training_label = training_label

    ## kNN Main Algorithm
    def __predictOne(self, testData):
        # Calculate distance
        distance = np.sum((testData - self.__training_data)**2, axis=1)**0.5
        # K Nearest Labels
        k_labels = [self.__training_label[i] for i in distance.argsort()[0:self.__k]]
        # Find the label with the largest amount
        label = Counter(k_labels).most_common(1)[0][0] # (label, amount)
        return label
    
    # Predict function for more than one row vector
    def predict(self, testing_data):
        if testing_data.ndim == 1:
            return self.__predictOne(testing_data)
        else:
            prediction = []
            for rowVector in testing_data:
                prediction.append(self.__predictOne(rowVector))
            return prediction
    
    # Calculate the accuracy of the testing dataset
    def score(self, testing_data, testing_label):
        predict_label = self.predict(testing_data)
        total = len(testing_label)
        correct = 0
        for i in range(total):
            if predict_label[i] == testing_label[i]:
                correct += 1
        return float(correct/total)

def loadData(path):
    letters = pd.read_csv(path)
    data = np.array(letters.drop(['lettr'], 1))
    label = np.array(letters['lettr'])
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainKNN(data_train, label_train, k):
    kNN = kNNClassifier(k)
    kNN.fit(data_train, label_train)
    return kNN

def testAccuracy(data_test, label_test, kNN):
    return kNN.score(data_test, label_test)

def evaluateModel(data_test, label_test, kNN):
    predict_label = kNN.predict(data_test)
    print(metrics.classification_report(label_test, predict_label))
    print(metrics.confusion_matrix(label_test, predict_label))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('kNN/kNN_Letter_Recognition/letter-recognition.csv')

    # Train Model
    kNN_model = trainKNN(data_train, label_train, 3)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, kNN_model)))

    # Evaluate Model
    evaluateModel(data_test, label_test, kNN_model)

if __name__ == '__main__':
    main()