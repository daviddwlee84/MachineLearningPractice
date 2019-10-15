# Linear Regression CSM Scikit Learn Version
#
# Author: David Lee
# Create Date: 2019/10/15
#
# Detail:
#   Total Data = 217
#   Training Data : Testing Data = 8 : 2

from tqdm import trange
import numpy as np
import pandas as pd  # Read csv

import torch

# Split training and testing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import metrics  # Evaluate model


def _numpyToTorch(X, y=None):
    X_torch = torch.from_numpy(X).type(torch.FloatTensor)
    if y is not None:
        y_torch = torch.from_numpy(y).type(torch.FloatTensor).unsqueeze(1)
        return X_torch, y_torch
    return X_torch


def loadData(path):
    inputData = pd.read_csv(path)
    # Conventional features
    conventionalFeatures = inputData[[
        'Genre', 'Gross', 'Budget', 'Screens', 'Sequel', 'Ratings']]
    # Gross income is not available before release
    newFeatures = conventionalFeatures.drop(['Gross'], 1)
    # Deal with missing value
    newFeatures = newFeatures.dropna()

    y = np.array(newFeatures['Ratings'])  # y
    X = np.array(newFeatures.drop(['Ratings'], 1))  # X

    # Normalize or the loss/gradient will become nan/inf!!
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=87)
    return X_train, y_train, X_test, y_test


def regression(X_train, y_train):
    learning_rate = 0.001
    X_train, y_train = _numpyToTorch(X_train, y_train)
    regression_model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 1))

    regression_model.train()

    optimizer = torch.optim.SGD(
        regression_model.parameters(), lr=learning_rate)

    t = trange(1000)
    for epoch in t:
        y_pred = regression_model(X_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train)
        t.set_postfix_str(f"epoch: {epoch+1}, loss: {loss.data.item()}")
        t.refresh()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return regression_model


def testAccuracy(X_test, y_test, regression_model):
    X_test, y_test = _numpyToTorch(X_test, y_test)
    regression_model.eval()
    y_pred = regression_model(X_test)
    accur2 = 0
    for i in range(len(y_test)):
        if abs(y_pred[i] - y_test[i]) <= 1:
            accur2 += 1
    accur2 /= len(y_test)

    print('Accuracy (Paper criteria Accuracy 2):', float(accur2))


def evaluateModel(X_test, y_test, regression_model):
    X_test, y_test = _numpyToTorch(X_test, y_test)
    regression_model.eval()
    y_pred = regression_model(X_test)

    # Mean Absolute Error (MAE)
    print('MAE:', metrics.mean_absolute_error(
        y_test.numpy(), y_pred.detach().numpy()))
    # Mean Squared Error (MSE)
    print('MSE:', metrics.mean_squared_error(
        y_test.numpy(), y_pred.detach().numpy()))
    # Root Mean Squared Error (RMSE)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(
        y_test.numpy(), y_pred.detach().numpy())))


def main():
    # Load Data
    X_train, y_train, X_test, y_test = loadData(
        'Datasets/2014-and-2015-CSM-dataset.csv')

    # Train Model
    regression_model = regression(X_train, y_train)

    # Test Accuracy
    testAccuracy(X_test, y_test, regression_model)

    # Evaluate Model
    evaluateModel(X_test, y_test, regression_model)


if __name__ == '__main__':
    main()
