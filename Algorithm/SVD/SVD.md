# Singular Value Decomposition

## Brief Description

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Unsupervised Learning|Dimensionality Reduction|SVD itself|Information Retrieval, Recommendation System

### Why SVD

If a is deficient and U is the computed echelon form (by Gaussian elimination), then, because of rounding errors in the elimination process, it is unlikely that U will have the proper number of nonzero rows.

PCA need the matrix to be square matrix, and if the matrix is too big, it will consume too many computing resource.

## Concept

### Approximate Original Matrix - Truncated SVD

We can use the **significant** singular value to reconstruct our original matrix

$$
A_{m\times n} \approx U_{m\times k} \Sigma_{k \times k} V^T_{k \times n}
$$

### Heuristics for the number of singular values to keep

#### Keep 90% of the energy expressed in the matrix

To calculate the total energy, you add up all the squared singular values. You can then add squared singular values until you reach 90% of the total.

#### Just guess a number

When the data matrix is too large, or you know your data well enough, you can make an assumption like this

### Transform to lower-dimensional space

Use U matrix to transform items data into the lower-dimensional space

$$
A^\mathrm{(Transformed)}_{n\times k} = A_{n\times m}^T U_{m\times k} \Sigma_{k\times k}
$$

## Mathematics Derivation

$$
A_{m\times n} = U_{m\times m} \Sigma_{m\times n} V^T_{n\times n}
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

### Numpy and Scipy

* [numpy.linalg.svd](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html)
`numpy.linalg.svd(a, full_matrices=True, compute_uv=True)`
> full_matrices : bool, optional
If True (default), u and vh have the shapes (..., M, M) and (..., N, N), respectively. Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N).
* [scipy.sparse.linalg.svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
    `scipy.sparse.linalg.svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)[source]`
    > k: Number of singular values and vectors to compute. Must be 1 <= k < min(A.shape)
    * [scipy.sparse.csc_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)

### Scikit Learn

* [Decomposition - Truncated singular value decomposition and latent semantic analysis](http://scikit-learn.org/stable/modules/decomposition.html#lsa)
    * [sklearn.decomposition.TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)
* [Cross decomposition](http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition)
    * [sklearn.cross_decomposition.PLSSVD](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD)

* [sklearn.utils.extmath.randomized_svd](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd)

### Gensim

* [models.lsimodel – Latent Semantic Indexing](https://radimrehurek.com/gensim/models/lsimodel.html) - Implements fast truncated SVD (Singular Value Decomposition)

### Article

* [CSDN - 數據預處理系列：（十二）用截斷奇異值分解降維](https://blog.csdn.net/u013719780/article/details/51767427)
