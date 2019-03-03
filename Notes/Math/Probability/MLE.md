# Maximum Likelihood Estimation

Used in *parametric estimation*.

* when you want to estimate something based on a hypothesized distribution
* it's easy to implement by using computer

## Basic

**Likelihood** of $q$ given the sample $X$

$$
l(\theta|X) = p(X|\theta) = \prod_t p(x^t|\theta)
$$

**Log likelihood**: (transfer product to sum)

$$
L(\theta|X) = \log l(\theta|X) = \sum_t \log p(x^t|\theta)
$$

**Masimum Likelihood Estimator (MLE)**

$$
\theta^* = \arg\max_\theta L(\theta|X)
$$

### Example of Bernulli Distribution

### Example of Multinomial Distribution

## Lagrange Multiplier

## Bias and Variance

For a **unknown parameter** $\theta$ and a **estimator** $d$ ($d(X)$ on sample $X$)

> $E[x]$ means the expected value of x

* Bias: $b_\theta(d) = E[d] - \theta$
* Variance: $E[(d-E[d])^2]$

### Mean Square Error

$$
r(d, \theta) = E[(d-\theta)^2] \\
= (E[d] - \theta)^2 + E[(d-E[d])^2] \\
= \text{Bias}^2 + \text{Variance}
$$

## Maximum a Posteriori Estimation

### Example: estimate mu

## Reference

* [wiki - Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

### Book

Deep Learning

* Maximum Likelihood Estimation - Ch 5.5

機器學習

* 極大似然估計 - Ch 7.2

Convex Optimization

* Least-Squares and Linear Programming - Ch 1.2
* Statistical Estimation - Ch 7
    * Maximum Likelihood Estimation - Ch 7.1.1
