## Gaussian Naive Bayes Nursery From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/9
#
# Detail:
#   Total Data = 12960
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

from collections import defaultdict # Auto-initialized dict
from operator import itemgetter # Find max in dictionary

import math # log

class GaussianNaiveBayesClassifier:
    def __init__(self, labels, mode='origin'):
        self.__labels = labels
        self.mode = mode # 'origin' & 'log'

    def fit(self, data_train, label_train):
        dataMat = np.mat(data_train)

        numTrain = len(label_train) # Total number of rows
        numAttr = len(data_train[0]) # Total number of columns

        attrProb = [] # 3D Table (but not in cube shape) of "each column's" "each tags's" "correcponding label's" probability
        for _ in range(numAttr):
            attrProb.append(defaultdict(dict))
        
        for col in range(numAttr): # Go through each attribute
            for label in self.__labels: # Calculate each label
                for i, val in enumerate(np.array(dataMat[:, col])): # Calculate the line fit the label, and record it
                    val = val[0] # Because val return [num] so use val[0] to get the num
                    # +1 in the attrProb Table when match the label
                    if label_train[i] == label:
                        # If not in the dict then initialize it
                        if label in attrProb[col][val]:
                            attrProb[col][val][label] += 1
                        else:
                            attrProb[col][val][label] = 1
                # Digits to percentage
                for key in attrProb[col].keys():
                    if label in attrProb[col][key]:
                        if self.mode == 'log':
                            attrProb[col][key][label] = -math.log(attrProb[col][key][label] / numTrain)
                        else:
                            attrProb[col][key][label] /= numTrain
                    else:
                        # Happends when some attribute don't have that label, it still need to be assigned to 0
                        attrProb[col][key][label] = 0
        self.attrProb = attrProb
    
    def __predictOne(self, data):
        probs = {}
        for label in self.__labels:
            probs[label] = 1
            for col, val in enumerate(data):
                if label in self.attrProb[col][val]:
                    ### DON'T KNOW WHY BUT IT CAN GET MORE ACCURACY IN LOG MODE
                    probs[label] *= self.attrProb[col][val][label]
                    # if self.mode == 'log':
                    #     probs[label] += self.attrProb[col][val][label]
                    # else:
                    #     probs[label] *= self.attrProb[col][val][label]
                else:
                    probs[label] = 0
        return max(probs.items(), key=itemgetter(1))[0] # Find label with max probability
      
    def predict(self, data_test):
        if data_test.ndim == 1:
            return self.__predictOne(data_test)
        else:
            prediction = []
            for rowVector in data_test:
                prediction.append(self.__predictOne(rowVector))
            return prediction

    def score(self, data_test, lable_test):
        label_predict = self.predict(data_test)
        total = len(lable_test)
        correct = 0
        for i in range(total):
            if label_predict[i] == lable_test[i]:
                correct += 1
        return float(correct/total)

def loadData(path):
    inputData = pd.read_csv(path)
    data = np.array(inputData.drop(['label'], 1))
    label = np.array(inputData['label'])
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainDecisionTree(data_train, label_train, mode):
    gnb = GaussianNaiveBayesClassifier(['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'], mode)
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
    GoussianNaiveBayesOrigin = trainDecisionTree(data_train, label_train, 'origin')
    GoussianNaiveBayesLog = trainDecisionTree(data_train, label_train, 'log')

    ## Origin
    # Test Accuracy
    print('Accuracy (origin):', float(testAccuracy(data_test, label_test, GoussianNaiveBayesOrigin)))
    # Evaluate Model
    evaluateModel(data_test, label_test, GoussianNaiveBayesOrigin)

    ## Log
    print('Accuracy (log):', float(testAccuracy(data_test, label_test, GoussianNaiveBayesLog)))
    evaluateModel(data_test, label_test, GoussianNaiveBayesLog)

if __name__ == '__main__':
    main()