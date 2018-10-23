## k-Means Sales Transactions Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/22
#
# Detail:
#   Total Data = 811

import numpy as np
import pandas as pd # Read csv
import matplotlib.pyplot as plt # Plot elbow

from sklearn.cluster import KMeans
from sklearn import metrics # Evaluate model


def loadData(path):
    inputData = pd.read_csv(path)
    inputData = inputData.drop(['Product_Code'], 1)
    data = np.array(inputData)
    return data

def trainKMeans(data_train, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_train)
    return kmeans

def testScore(data_train, kmeans):
    return kmeans.score(data_train)

def evaluateModel(data_train, kmeans):
    labels = kmeans.labels_
    silhouette_score = metrics.silhouette_score(data_train, labels, metric="euclidean")
    calinski_harabaz_score =  metrics.calinski_harabaz_score(data_train, labels)
    print("the mean Silhouette Coefficient of all samples:", silhouette_score)
    print("the Calinski and Harabaz score:", calinski_harabaz_score)
    return silhouette_score, calinski_harabaz_score

def main():

    MAX_TRY = 20

    # Load Data
    data_train = loadData('Datasets/Sales_Transactions_Dataset_Weekly.csv')

    # Train Model and Test Score and Evaluate Model
    # try many different k
    scores = []
    scores2 = []
    scores3 = []
    for k in range(2, MAX_TRY+1):
        # Train
        kMeans_model = trainKMeans(data_train, k)
        # Score
        score = float(testScore(data_train, kMeans_model))
        scores.append(score)
        print('Score of k = %d:' % (k), score)
        # Evaluate
        score2, score3 = evaluateModel(data_train, kMeans_model)
        scores2.append(score2)
        scores3.append(score3)

    # Plot the k - loss diagram
    fig = plt.figure(1, figsize=(15, 5))
    fig.suptitle('Comparison of three metrics score')

    plt.subplot(131)
    plt.ylabel("Opposite of the value of X on the K-means objective")
    plt.grid(True)
    plt.xticks(range(2, MAX_TRY+1))
    plt.plot(np.arange(2, MAX_TRY+1, 1), np.array(scores)*-1)

    plt.subplot(132)
    plt.ylabel("Mean Silhouette Coefficient of all samples")
    plt.grid(True)
    plt.xticks(range(2, MAX_TRY+1))
    plt.plot(np.arange(2, MAX_TRY+1, 1), np.array(scores2))

    plt.subplot(133)
    plt.ylabel("Calinski and Harabaz score")
    plt.grid(True)
    plt.xticks(range(2, MAX_TRY+1))
    plt.plot(np.arange(2, MAX_TRY+1, 1), np.array(scores3))
    k_max = np.arange(2, MAX_TRY+1, 1)[scores3.index(max(scores3))]
    arrow_xy = [k_max, max(scores3)]
    text_xy = arrow_xy[:]
    text_xy[0] += MAX_TRY//2
    text_xy[1] = text_xy[1] * 0.8
    plt.annotate('k of max\n(k=%d)' % (k_max), xy=arrow_xy, xytext=text_xy,
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.98, hspace=0.20,
                    wspace=0.20)
    plt.show()

if __name__ == '__main__':
    main()