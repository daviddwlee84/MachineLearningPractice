# k-means

## Brief Description

k-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Unsupervised Learning|Clustering||

### Steps

1. Initialize the center of the clusters (pick centroids)
2. Attribute the closest cluster to each data point (calculate distance)
3. Set the position of each cluster to the mean of all data points belonging to that cluster (move to new cluster center)
4. Repeat STEP 2-3 until convergence

### Deciding umber of cluster

* An incorrect choice of the number of clusters will invalidate the whold process
* Try k-means clustering with different number of clusters and measure the resulting sum of squares

### When to use k-means clustering

* Best used when the number of cluster centers is specified due to a well-defined list of types shown in the data

* Don't use
    * Overlapping data => Euclidean distance doesn't measure that underlined in fact as well
    * If data is noisy or full of outliers

### Pros and Cons

* Advantages
    * K-means speed > hierarchical clusterning speed (if k is small)
    * K-means may produce tighter clusters than hierarchical clustering
* Disadvantages
    * Difficulty in comparing quality of the clusters produced
    * Fixed number of clusters can make it difficult to predict what k should be (e.g. K-means Trap)
    * Strong sensitivity to outliers and noise
    * Doesn't work well with non-circular cluster shape (Non-globular cluster)
    * Low capability to pass the local optimum

## Terminology

* Centroid
* Elbow method - find the best K

## Concepts

### Centroid

### Distance

#### Euclidean distance

#### Distance Calculation Library

* `numpy.linalg.norm(a-b)`
* `scipy.spatial.distance.euclidean(a, b)`

## Links

### Tutorial

* [Youtube - K-Means Clustering](https://youtu.be/3vHqmPF4VBA)
* [Siraj Raval - K-Means Clustering](https://youtu.be/9991JlKnFmk)
    * [Code](https://github.com/llSourcell/k_means_clustering)

### Wikipedia

* [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)

### Scikit Learn

* [Example of K-means Clustering](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py)
* [Clustering](http://scikit-learn.org/stable/modules/clustering.html)
* [sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)
