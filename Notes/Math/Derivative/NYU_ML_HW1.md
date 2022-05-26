# NYU Machine Learning HW1

## 1

Let $\{x_1, x_2, \dots, x_n\}$ be a set of points in $d$-dimentional space. Suppose we wish to produce a single point estimate $\mu \in \mathcal{R}^d$ that minimizes the mean squared-error:

$$
\frac{1}{n} (||x_1 - \mu||^2_2 + ||x_2 - \mu||^2_2 + \dots + ||x_n - \mu||^2_2)
$$

Find a closed form expression for $\mu$ and prove that your answer is correct.

### Solution

> [Proof (part 1) minimizing squared error to regression line | Khan Academy - YouTube](https://www.youtube.com/watch?v=mIx2Oj5y9Q8)

## 2

Not all norms behave the same; for instance, the $l_1$-norm of a vector can be dramatically different from the $l_2$-norm, especially in high dimensions. Prove the following norm inequalities for $d$-dimensional vectors, starting from the definitions provided in class and lecture notes. (Use any algebraic technique/result you like, as long as you cite it.)

1. $||x||_\infty \leq ||x||_2 \leq \sqrt{d}||x||_\infty$
2. $||x||_\infty \leq ||x||_1 \leq d||x||_\infty$

> [analysis - 1 and 2 norm inequality - Mathematics Stack Exchange](https://math.stackexchange.com/questions/2293778/1-and-2-norm-inequality)

---

p-norm

infinity norm
