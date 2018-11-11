# Information Retrieval

## Latent Semantic Analysis (Latent Semantic Indexing)

Dimensionality reduction using truncated SVD (aka LSA)

This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently.

In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).

This estimator supports two algorithms: a fast randomized SVD solver, and a “naive” algorithm that uses ARPACK as an eigensolver on (X * X.T) or (X.T * X), whichever is more efficient.

## Links

* [Wiki - Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)