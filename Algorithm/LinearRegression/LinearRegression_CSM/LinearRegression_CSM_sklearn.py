## Linear Regression CSM Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/8
#
# Detail:
#   Total Data = 217
#   Training Data : Testing Data = 8 : 2

import numpy as np
import pandas as pd # Read csv

from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer # Scikit Learn v0.20
# from sklearn.preprocessing import imputation # Scikit Learn v0.19
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

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
    print('R2:', float(regression_model.score(X_test, y_test)))
    total_predict = len(y_test)
    y_pred = regression_model.predict(X_test)
    accur2 = 0
    for i in range(total_predict):
        if abs(y_pred[i] - y_test[i]) <= 1:
            accur2 += 1
    accur2 /= total_predict

    print('Accuracy (Paper criteria Accuracy 2):', float(accur2))

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

if __name__ == '__main__':
    main()