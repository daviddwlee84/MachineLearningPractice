# Linear Algebra Basis

## Definitions

* Inner products: $\langle x, y \rangle = x^Ty$
* Norm
  * L2 Norm: $||x||^2 = x^Tx$
* Distance:
  * $d(x, y) = ||x - y||$
  * $d_M(x, y) = \sqrt{(x-y)^T M(x-y)}$
* Orthogonality $\langle x, y \rangle = 0$

Needed to define norm preserving (i.e. orthogonal) transforms

* Similarity between two vectors: $|\langle x, y \rangle| = |x||y|\cos\theta$
  * maximum when the two vector are aligned
  * minimum when they are orthogonal to each other
* Transforming a vector
  * $y = Tx$

## Subspaces and bases

Let $S\in R^N$ be a subset of vectors in $R^N$

### Span

### Bases

## Eigenvectors and Eigenvalues

...

$$
x^TLx = \sum (x_i - x_j)^2 = x^T(U\Lambda U^T)x \\
= (U^Tx)\Lambda(U^Tx) = \alpha^T \Lambda \alpha
= \sum_{k=1}^N \lambda_k\alpha_k^2
$$

> total variation

## Resources

* [**Immersive Math - Linear Algebra**](http://immersivemath.com/ila/index.html)
* [Essence of Linear Algebra - YouTube](https://www.youtube.com/playlist?list=PL_w8oSr1JpVCZ5pKXHKz6PkjGCbPbSBYv)
