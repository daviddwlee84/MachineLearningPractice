## k-Means Sales Transactions From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/23
#
# Detail:
#   Total Data = 811

import numpy as np
import pandas as pd # Read csv
import matplotlib.pyplot as plt # Plot elbow

from sklearn import metrics # Evaluate model

# Current support Euclidian distance
# TODO: support other distances, score method
class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=0.0001):
        self.__k = n_clusters
        self.__max_iter = max_iter
        self.__tol = tol
    
    # Initialize the centroids as k random sample of X
    def __initRandomCentroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.__k, n_features))
        for i in range(self.__k):
            centroid = X[np.random.choice(range(n_samples))] # Just randomly pick one sample as centroid
            centroids[i] = centroid
        return centroids
    
    def __euclideanDistance(self, a, b):
        return np.linalg.norm(a-b)

    # Return the index of the cloest centroid to the sample by euclidean distance
    def __closestCentroid(self, sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = self.__euclideanDistance(sample, centroid) # Calculate distance
            # if found the minimum distance than update
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i
    
    # Assign the samples to the closest centroids to create clusters (a list)
    def __createClusters(self, centroids, X):
        clusters = [[] for _ in range(self.__k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.__closestCentroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters
    
    # Calculate new centroids as the means of the samples in each cluster
    def __calculateNewCentroids(self, clusters, X):
        n_features = np.shape(X)[1]
        newCentroids = np.zeros((self.__k, n_features))
        for i, cluster in enumerate(clusters):
            # Use cluster index to indicate samples and calculate mean of each row
            centroid = np.mean(X[cluster], axis=0)
            newCentroids[i] = centroid
        return newCentroids
        
    # Classify samples as the index of their clusters
    def __getClusterLabels(self, clusters, X):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Calculate final clusters
    def fit(self, X):
        # Initialize centroids as k random samples from X
        centroids = self.__initRandomCentroids(X)

        iteration = 0
        # Iterate until convergence
        while iteration < self.__max_iter:
            # Assign samples to closest centroids (create clusters)
            clusters = self.__createClusters(centroids, X)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self.__calculateNewCentroids(clusters, X)
            # Check if all the value in diff (2D array) is less than tolerance => convergence
            centroid_diff = abs(centroids - prev_centroids)
            if all(all(single_val < self.__tol for single_val in a_centroid) for a_centroid in centroid_diff):
                break
            iteration += 1
        
        self.__clusters = clusters
        self.cluster_centers_ = centroids
        self.labels_ = self.__getClusterLabels(self.__clusters, X)

    # Find the cloeset centroid of unseen data
    def predict(self, X):
        # If only one row of data (i.e. Dimension = 1)
        if X.ndim == 1:
            return self.__closestCentroid(X, self.cluster_centers_)
        else:
            prediction = np.zeros(np.shape(X)[0])
            for i, row in enumerate(X):
                prediction[i] = self.__closestCentroid(row, self.cluster_centers_)
            return prediction

def loadData(path):
    inputData = pd.read_csv(path)
    inputData = inputData.drop(['Product_Code'], 1) # Drop prodoct ID column
    data = np.array(inputData)
    return data

def trainKMeans(data_train, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_train)
    return kmeans

def testPredict(data_train):
    # Test a single line prediction of KMeans model with k = 3
    testdata = [94,169,8,9,10,8,7,13,12,6,14,9,4,7,12,8,7,11,10,7,7,13,11,8,10,8,14,5,3,13,11,9,7,8,7,9,6,12,12,9,3,5,6,14,5,5,7,8,14,8,8,7,3,14,0.36,0.73,0.45,0.55,0.64,0.45,0.36,0.91,0.82,0.27,1.00,0.55,0.09,0.36,0.82,0.45,0.36,0.73,0.64,0.36,0.36,0.91,0.73,0.45,0.64,0.45,1.00,0.18,0.00,0.91,0.73,0.55,0.36,0.45,0.36,0.55,0.27,0.82,0.82,0.55,0.00,0.18,0.27,1.00,0.18,0.18,0.36,0.45,1.00,0.45,0.45,0.36]
    testdata2 = [17,4,45,36,31,28,28,34,42,40,43,35,30,33,40,45,48,35,30,29,39,41,30,34,28,22,31,29,30,34,26,28,32,31,27,40,33,34,31,19,48,21,31,19,24,32,36,30,33,29,27,29,19,48,0.24,0.41,0.90,0.59,0.41,0.31,0.31,0.52,0.79,0.72,0.83,0.55,0.38,0.48,0.72,0.90,1.00,0.55,0.38,0.34,0.69,0.76,0.38,0.52,0.31,0.10,0.41,0.34,0.38,0.52,0.24,0.31,0.45,0.41,0.28,0.72,0.48,0.52,0.41,0.00,1.00,0.07,0.41,0.00,0.17,0.45,0.59,0.38,0.48,0.34,0.28,0.34]
    testdatas = [testdata, testdata2]

    kMeans_model = trainKMeans(data_train, 3)
    # Test single data
    print(kMeans_model.predict(np.array(testdata)))
    # Test multiple data
    print(kMeans_model.predict(np.array(testdatas)))

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

    #testPredict(data_train)

    # Train Model and Evaluate Model
    # try many different k
    scores2 = []
    scores3 = []
    for k in range(2, MAX_TRY+1):
        # Train
        kMeans_model = trainKMeans(data_train, k)
        print('Score of k = %d:' % (k))
        # Evaluate
        score2, score3 = evaluateModel(data_train, kMeans_model)
        scores2.append(score2)
        scores3.append(score3)

    # Plot the k - loss diagram
    fig = plt.figure(1, figsize=(10, 5))
    fig.suptitle('Comparison of three metrics score')

    plt.subplot(121)
    plt.ylabel("Mean Silhouette Coefficient of all samples")
    plt.grid(True)
    plt.xticks(range(2, MAX_TRY+1))
    plt.plot(np.arange(2, MAX_TRY+1, 1), np.array(scores2))

    plt.subplot(122)
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