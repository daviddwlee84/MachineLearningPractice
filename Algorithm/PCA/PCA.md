# Principal Compnent Analysis

## Brief Description

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

or

A method for doing dimensionality reduction by transforming the feature space to a lower dimensionality, removing correlation between features and maximizing the variance along each feature axis.

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Unsupervised Learning|Dimensionality Reduction|Orthogonal, Covariance Matrix, Eigenvalue Analysis|

## Concepts

### PCA Algorithm

Steps

* Take the first principal component to be in the direction of the largest variability of the data
* The second preincipal component will be in the direction orthogonal to the first principal component
> (We can get these values by taking the covariance matrix of the dataset and doing eigenvalue analysis on the covariance matrix)
* Once we have the eigenvectors of the covariance matrix, we can take the top N eigenvectors => N most important feature
* Multiply the data by the top N eigenvectors to transform our data into the new space

Pseudocode

```
Remove the mean
Compute the covariance matrix
Find the eigenvalues and eigenvectors of the covariance matrix
Sort the eigenvalues from largest to smallest
Take the top N eigenvectors
Transform the data into the new space created by the top N eigenvectors
```

## Deduction

Variables

* m x n matrix: $X$
    * In practice, column vectors of $X$ are positively correlated
    * the hypothetical factors that account for the score should be uncorrelated
* orthogonal vectors: $\vec{y}_1, \vec{y}_2, \dots, \vec{y}_r$
    * We require that the vectors span $R(X)$
    * and hence the number of vectors, $r$, should be euqal to the rank of $X$

The covariance matrix is
$$
S = \frac{1}{n-1} X^TX
$$

**The first principal component vector**, $\vec{y}_1$ should account for the most variance. (Since $\vec{y}_1$ is in the column space of $X$, we can represent it as a product $X\vec{v}_1$)
$$
\mathrm{var}(\vec{y}_1) = \frac{(X\vec{v}_1)^T(X\vec{v}_1)}{n-1} = \vec{v}_1^TS\vec{v}_1
$$

The vector $\vec{v}_1$ is chosen to maximize $\vec{v}^TS\vec{v}$ over all unit vectors $\vec{v}$

=> Choosing $\vec{v}_1$ to be a unit eigenvector of $X^TX$ belonging to its maximum eigenvalue $\lambda_1$

The eigenvectors of $X^TX$ are the right singular vectors of X. Thus,  $\vec{v}_1$ is the right singular vector of X correspondig to the largest singular value $\sigma_1 = \sqrt{\lambda_1}$

If $\vec{u}_1$ is the corresonding left singular vector, then
$$
\vec{y}_1 = X\vec{v}_1 = \sigma_1\vec{u}_1
$$

**The second principal component vector** must be of the form $\vec{y}_2 = X\vec{v}_2$.

It can be shown that the vector which maximizes $\vec{v}^TS\vec{v}$ over all unit vectors that are orthogonal to $\vec{v}_1$ is just the second right singular vector $\vec{v}_2$ of $X$.

If we choose $\vec{v}_2$ in this way and $\vec{u}_2$ is the corresponding left singular vector, then
$$
\vec{y}_2 = X\vec{v}_2 = \sigma_2\vec{u}_2
$$
and since
$$
\vec{y}_1^T\vec{y}_2 = \sigma_1\sigma_2\vec{u}_1^T\vec{u}_2 = 0
$$
it follows that $\vec{y_1}$ and $\vec{y_2}$ are orthogonal.

(The remaining $\vec{y}_i$'s are determined in a similar manner)

## Reference

* Linear Algebra with Applications
    * Ch 5 Orthogonality
    * Ch 6 Eigenvalues
    * Ch 6.5 Application 4 - PCA
    * Ch 7.5 Orthogonal Transformations
    * Ch 7.6 The Eigenvalue Problem

## Links

[Principal component analysis - Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)

### Scikit Learn

* [Decomposing signals in components (matrix factorization problems)](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)
* [sklearn.decomposition](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition)
* [sklearn.decomposition.PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
