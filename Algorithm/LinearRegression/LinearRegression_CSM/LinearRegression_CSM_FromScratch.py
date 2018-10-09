## Linear Regression CSM Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/9
#
# Detail:
#   Total Data = 217
#   Training Data : Testing Data = 8 : 2

import numpy as np
import pandas as pd # Read csv

from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

class LinearRegression:

    # Calculate weights
    def __calcWeights(self, xArr, yArr):
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        xTx = xMat.T * xMat
        if np.linalg.det(xTx) == 0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * yMat)
        return ws

    def rssError(self, yArr, yHatArr):
        return ((yArr - yHatArr)**2).sum()

    def fit(self, X, y):
        self.weights = self.__calcWeights(X, y)

    def __predictOne(self, x):
        yHat = np.mat(x) * self.weights
        return yHat

    def predict(self, X_test):
        m = np.shape(X_test)[0]
        if m == 1:
            return self.__predictOne(X_test)
        else:
            prediction = np.zeros(m)
            for i, rowVector in enumerate(X_test):
                prediction[i] = self.__predictOne(rowVector)
            return prediction
    
    def score(self, X_test, y_test, method='diff', diff=1):
        if method == 'diff':
            yHats = self.predict(X_test)
            total = len(y_test)
            inRange = 0
            for i in range(total):
                if abs(yHats[i] - y_test[i]) <= diff:
                    inRange += 1
            return float(inRange/total)

def loadData(path):
    inputData = pd.read_csv(path)
    # Conventional features
    conventionalFeatures = inputData[['Genre', 'Gross', 'Budget', 'Screens', 'Sequel', 'Ratings']]
    # Gross income is not available before release
    newFeatures = conventionalFeatures.drop(['Gross'], 1)
    # Deal with missing value
    newFeatures = newFeatures.dropna()

    y = np.array(newFeatures['Ratings']) # y
    X = np.array(newFeatures.drop(['Ratings'], 1)) # X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)
    return X_train, y_train, X_test, y_test
    
def regression(X_train, y_train):
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    return regression_model

def testAccuracy(X_test, y_test, regression_model):
    print('Accuracy (Paper criteria Accuracy 2):', float(regression_model.score(X_test, y_test, 'diff', 1)))

def evaluateModel(X_test, y_test, regression_model):
    # Mean Absolute Error (MAE)
    print('MAE:', metrics.mean_absolute_error(y_test, regression_model.predict(X_test)))
    # Mean Squared Error (MSE)
    print('MSE:', metrics.mean_squared_error(y_test, regression_model.predict(X_test)))
    # Root Mean Squared Error (RMSE)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, regression_model.predict(X_test))))

def main():
    # Load Data
    X_train, y_train, X_test, y_test = loadData('Datasets/2014-and-2015-CSM-dataset.csv')

    # Train Model
    regression_model = regression(X_train, y_train)

    # Test Accuracy
    testAccuracy(X_test, y_test, regression_model)

    # Evaluate Model
    evaluateModel(X_test, y_test, regression_model)
    print('RSS:', regression_model.rssError(y_test, regression_model.predict(X_test)))

if __name__ == '__main__':
    main()