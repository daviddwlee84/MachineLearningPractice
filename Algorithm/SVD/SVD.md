# Singular Value Decomposition

## Brief Description

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Unsupervised Learning|Dimensionality Reduction|SVD itself|Information Retrieval, Recommendation System

### Why SVD

If a is deficient and U is the computed echelon form (by Gaussian elimination), then, because of rounding errors in the elimination process, it is unlikely that U will have the proper number of nonzero rows.

## Mathematics Deduction

$$
A = U\Sigma V^T
$$

Material

* $A$ : an $m\times n$ (rank-deficient) matrix
* $U$ : an $m\times m$ orthogonal matrix
* $V$ : an $n\times n$ orthogonal matrix
* $\Sigma$ : an $m\times n$ matrix
    * whose off diagonal entries are all 0's
    * whose diagonal elements satisfy
        $$
        \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0\\
        \Sigma = \begin{bmatrix}
        \sigma_1 \\
        & \sigma_2 \\
        && \ddots \\
        &&& \sigma_n \\
        & \\
        & \\
        \end{bmatrix}
        $$
* The $\sigma_i$'s determined by this factorization are unique and are called the *singular values* of $A$
* The factorization $U\Sigma V^T$ is called the *singular value decomposition* of $A$
    * The rank of $A$ equals the number of nonzero singular values
    * The magnitudes of the nonzero singular values provide a measure of how close $A$ is to a matrix of lower rank

## Reference

* Linear Algebra with Applications
    * Ch 6.5 The Singular Value Decomposition

## Links

* [Wiki - Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)

### Tutorial

* [MIT - Singular Value Decomposition (the SVD)](https://www.youtube.com/watch?v=mBcLRGuAFUk)
* [MIT - Computing the Singular Value Decomposition](https://www.youtube.com/watch?v=cOUTpqlX-Xs)

* [Stanford Andrew Ng - Lecture 16.5 — Recommender Systems | Vectorization Low Rank Matrix Factorization](https://www.youtube.com/watch?v=5R1xOJOFRzs)

* [Stanford - Lecture 46 — Dimensionality Reduction - Introduction](https://youtu.be/yLdOS6xyM_Q)
* [Stanford - Lecture 47 — Singular Value Decomposition](https://youtu.be/P5mlg91as1c)
* [Stanford - Lecture 48 — Dimensionality Reduction with SVD](https://www.youtube.com/watch?v=UyAfmAZU_WI)
* [Stanford - Lecture 49 — SVD Gives the Best Low Rank Approximation (Advanced)](https://youtu.be/c7e-D2tmRE0)
* [Stanford - Lecture 50 — SVD Example and Conclusion](https://youtu.be/K38wVcdNuFc)

### Scikit Learn

* [Cross decomposition](http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition)
    * [sklearn.cross_decomposition.PLSSVD](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD)
* [Decomposition - Truncated singular value decomposition and latent semantic analysis](http://scikit-learn.org/stable/modules/decomposition.html#lsa)
    * [sklearn.decomposition.TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)

* [sklearn.utils.extmath.randomized_svd](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd)
