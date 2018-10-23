## k-Means Sales Transactions Scikit Learn Version
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
        self.labels_ = self.predict(X)
    
    # Classify samples as the index of their clusters
    def __getClusterLabels(self, clusters, X):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        return self.__getClusterLabels(self.__clusters, X)
        

def loadData(path):
    inputData = pd.read_csv(path)
    inputData = inputData.drop(['Product_Code'], 1)
    data = np.array(inputData)
    return data

def trainKMeans(data_train, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_train)
    return kmeans

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

    # Plot the k - loss diagram (Elbow Method)
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