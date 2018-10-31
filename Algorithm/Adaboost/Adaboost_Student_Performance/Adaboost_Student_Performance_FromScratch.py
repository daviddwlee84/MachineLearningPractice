## Student Performace From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/31
#
# Detail:
#   Total Data = 649
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

# A simple decision tree as a weak learner
class DecisionStump:
    def __init__(self):
        self.polarity = 1 # Determines if sample shall be classified as -1 or 1 given threshold
        self.feature_index = None # The index of the feature used to make classification
        self.threshold = None # The threshold value that the feature should be measured against
        self.alpha = None # Value indicative of the classifier's accuracy

class AdaBoostClassifier:
    def __init__(self, n_clf=5):
        self.__n_clf = 5 # The number of weak classifiers that will be used

    def __calcAlpha(self, error):
        return 0.5 * np.log((1.0 - error) / error + 1e-10)
    
    def __buildStump(self, X, y):
        n_feature = np.shape(X)[1]

        clf = DecisionStump()

        min_error = float('inf')

        # Iterate through every unique feature value and see what value makes the best threshold for predicting y
        for feature_i in range(n_feature):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)

            # Try every unique feature value as threshold
            for threshold in unique_values:
                polarity = 1

                prediction = np.ones(np.shape(y)) # Set all predictions to '1' initially
                prediction[X[:, feature_i] < threshold] = -1 # Label the samples whose values are below threshold as '-1

                error = sum(self.__D[y != prediction]) # Sum of weight D of misclassified samples

                # If the error is over 50% we flip the polarity (inverse the prediction e.g. 0 to 1)
                if error > 0.5:
                    error = 1 - error
                    polarity = -1

                # If this threshold resulted in the smallest error, we save the configuration
                if error < min_error:
                    clf.polarity = polarity
                    clf.threshold = threshold
                    clf.feature_index = feature_i
                    min_error = error # Update min error

        # Calculate the alpha
        clf.alpha = self.__calcAlpha(min_error)

        final_predictions = np.ones(np.shape(y))
        negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
        final_predictions[negative_idx] = -1

        # Claculate new weight D
        self.__D *= np.exp(clf.alpha * -y * final_predictions)
        self.__D /= np.sum(self.__D) # Normalize to one

        return clf

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples = np.shape(X)[0]
        self.__D = np.full((n_samples, 1), (1/n_samples)) # Initialize weight in same value

        self.__decisionStumps = [] # Save classifiers

        # Iterate through classifiers
        for _ in range(self.__n_clf):
            stump = self.__buildStump(X, y)
            self.__decisionStumps.append(stump)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))

        # For each classifier => label the samples
        for clf in self.__decisionStumps:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            y_pred += clf.alpha * predictions
        
        y_pred = np.sign(y_pred).flatten() # Return sign of prediction sum
        y_pred = np.mat(y_pred).T
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return sum(y_pred == y_test) / len(y_test)

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
        if finalGrade >= 10:
            label[i] = 1 # Pass
        else:
            label[i] = -1 # Fail

    data_train, data_test, label_train, label_test = train_test_split(dataTemp, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainAdaboost(data_train, label_train):
    clf = AdaBoostClassifier(n_clf=16)
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