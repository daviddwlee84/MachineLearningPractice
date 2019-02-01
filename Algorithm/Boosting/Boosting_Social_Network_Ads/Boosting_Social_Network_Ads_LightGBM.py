## LightGBM Social Network Ads
#
# Author: David Lee
# Create Date: 2019/1/22

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split # For splitting trian and test set
from sklearn.preprocessing import StandardScaler # For feature scaling

import lightgbm as lgb

from sklearn.metrics import confusion_matrix, accuracy_score

params = {
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 10,
    'min_data': 50,
    'max_depth': 10,
    'boost_from_average': False
}

def data_preprocessing(path):
    # Importing the dataset
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, [2, 3]].values # Age, Estimated Salary
    y = dataset.iloc[:, 4].values # Purchased

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test= sc.transform(x_test)

    return x_train, x_test, y_train, y_test

def model_building_training(x_train, y_train):
    d_train = lgb.Dataset(x_train, label=y_train)
    clf = lgb.train(params, d_train, 100)
    return clf

def predict(x_test, clf):
    y_pred = clf.predict(x_test)
    
    # Convert into binary values
    y_pred[y_pred >=0.5] = 1
    y_pred[y_pred < 0.5] = 0

    return y_pred

def evaluate(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy:', accuracy)

def main():
    x_train, x_test, y_train, y_test = data_preprocessing('Datasets/Social_Network_Ads.csv')
    clf = model_building_training(x_train, y_train)
    y_pred = predict(x_test, clf)
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()
