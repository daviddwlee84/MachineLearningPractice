# Linear Regression Derivation

## Linear Regression

The dataset is made of $m$ training examples $(x(i), y(i))_{i \in[m]},$ where $x(i) \in \mathbb{R}^{d}$ are the features
and $y(i) \in \mathbb{R}$ are the target variables.
Assumption, there exists $\theta \in \mathbb{R}^{d}$ such that:
$$
y(i)=\theta^{T} x(i)+\epsilon(i)
$$
with $\epsilon(i)$ i.i.d. Gaussian random variables (mean zero and variance $\sigma^{2}$)

Computing the likelihood function gives:
$$
\begin{aligned} L(\theta) &=\prod_{i=1}^{m} p_{\theta}(y(i) | x(i)) \\
\quad&=\prod_{i=1}^{m} \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{\left(y(i)-\theta^{T} x(i)\right)^{2}}{2 \sigma^{2}}\right)
\end{aligned}
$$

Maximizing the log likelihood:
$$
\begin{aligned} \ell(\theta) &=\log L(\theta) \\ &=-m \log (\sigma \sqrt{2 \pi})-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{m}\left(y(i)-\theta^{T} x(i)\right)^{2}
\end{aligned}
$$

is the same as minimizing the cost function:
$$
J(\theta)=\frac{1}{2} \sum_{i=1}^{m}\left(y(i)-\theta^{T} x(i)\right)^{2}
$$
giving rise to the ordinary least squares regression model.

The gradient of the least-squares cost function is:
$$
\begin{aligned} \frac{\partial}{\partial \theta_{j}} J(\theta)&=\sum_{i=0}^{m}\left(y(i)-\theta^{T} x(i)\right) \frac{\partial}{\partial \theta_{j}}\left(y(i)-\sum_{k=0}^{d} \theta_{k} x_{k}(i)\right)\\&=\sum_{i=0}^{m}\left(y(i)-\theta^{T} x(i)\right) x_{j}(i)
\end{aligned}
$$

## Gradient Descent Algorithm

Batch gradient descent perfoms the update:
$$
\theta_{j}:=\theta_{j}+\alpha \sum_{i=0}^{m}\left(y(i)-\theta^{T} x(i)\right) x_{j}(i)
$$

where $\alpha$ is the learning rate.

This method looks at every example in the entire training set on every step.

Stochastic gradient descent very well. The sum above is 'replaced' by a loop over the training examples, so that the update becomes:

for $i=1$ to $m$ :
$$
\theta_{j}:=\theta_{j}+\alpha\left(y(i)-\theta^{T} x(i)\right) x_{j}(i)
$$

## Back to Linear Regression

Recall that under mild assumptions, the explicit solution for the ordinary least squares can be
written explicitely as:

$$
\theta^{*} = (X^{T} X)^{-1} X^{T} Y
$$

where the linear model is written in matrix form $Y=X \theta+\epsilon$, with $Y=(y(1), \ldots y(m)) \in \mathbb{R}^{m}$ and $X=(x(1), \ldots x(m)) \in \mathbb{R}^{m \times d}$

> Exercise: show the above formula is valid as soon as $\operatorname{rk}(X)=d$

In other words, **optimization is useless for linear regression**! (if you are able to invert $X^TX$)

Nevertheless, gradient descent algorithms are important optimization algorithms with theoretical guarantees when the cost function is convex.

## References

* [Regressions, Classification and PyTorch Basics](https://mlelarge.github.io/dataflowr-slides/X/lesson2.html)
* [The Principle of Maximum Likelihood - Linear Regression : The Probabilistic Perspective](http://complx.me/2017-01-22-mle-linear-regression/)
