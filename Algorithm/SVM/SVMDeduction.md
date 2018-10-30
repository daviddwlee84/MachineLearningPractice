# SVM Mathematics Deduction

## Overview

### Phases

* Linear Seprable SVM  <- Data is linear seprable (the simplest condition)
* Linear SVM           <- Data has a little *noise* (Not entirely linear seprable)
* Gaussian Kernel SVM  <- Data is too complicated

### Optimization

* Lagrange
* SMO

### Theorem

* Lagrange Duality
    * Karush-Kuhn-Tucker (KKT)

### Big Picture

1. [Original Problem](#Original-Problem) $(w, b)$ $\xRightarrow{\text{Lagrange Multiplier}}$ [Dual Problem](#Dual-Algorithm) $(w, b)$ $\xRightarrow{KKT}$ Solve alpha [Use SMO Here]
2. Solved alpha (you'll know which are support vectors) $\Rightarrow$ Find w, b (i.e. found separating hyperplane)

## Assume data is linear seprable

### Original Problem

Testing Dataset:
$$T = \{(\vec{x}_1, y_1), (\vec{x}_2, y_2), \dots, (\vec{x}_N, y_N)\}$$
Conditions: $\vec{x}_i \in X \in \mathbb{R}^n$, $y_i \in y = (+1, -1)$, $i = 1, \dots, N$

> $\vec{x}_i$ is ith data feature vector, $y_i$ is labeled class of $\vec{x}_i$

### Goal: Definition of Linear Seprable SVM

Find a separating hyperplane:
$$\vec{w} \cdot \vec{x} + b = 0$$

Classification function:
$$f(x) = \mathrm{sign}(\vec{w} \cdot \vec{x} + b)$$

---

### Margin

Distance between a vector x and the separating hyperplane:
$$|\vec{w} \cdot \vec{x} + b|$$

We can represent our correctness of classification by:
$$y_i (\vec{w} \cdot \vec{x} + b)$$

* If classify correctly => $y_i$ and $(\vec{w} \cdot \vec{x} + b)$ will have same sign => Positive product
* Else => Negative product

#### Functional Margin

Based on testing data set T and separating hyperplane (w, b)
Define functional margin between separating hyperplane (w, b) and data points:
$$ \hat{\gamma_i} = y_i(\vec{w} \cdot \vec{x_i} + b)$$

We'd like to find the point *closest to the separating hyperplane* and make sure this is *as far away from the separating line* as possible

$$ \hat{\gamma} = \min_{i= 1,\dots,N} \hat{\gamma_i}$$

#### Geometric Margin

$$ \gamma_i = y_i \bigg(\frac{\vec{w}}{||\vec{w}||} \cdot \vec{x_i} + \frac{b}{||\vec{w}||}\bigg) $$

#### Relationship of Functional Margin and Geometric Margin

$$ r = \frac{\hat{r}}{||\vec{w}||} $$

when ||w|| = 1 <=> functional margin = geometric margin

### Maximize Margin

Optimization Problem

$$
\max_{\vec{w}, b} \gamma \\

\ni y_i(\frac{\vec{w}}{||\vec{w}||} \cdot \vec{x_i} + \frac{b}{||\vec{w}||}) \geq \gamma
$$

$\gamma \rightarrow \frac{\hat{\gamma}}{||\vec{w}||}$

$$
\max_{\vec{w}, b} \frac{\hat{\gamma}}{||\vec{w}||} \\

\ni y_i(\vec{w} \cdot \vec{x_i} + b) \geq \hat{\gamma}
$$

Let $\hat{\gamma} = 1$

And we found that $\max_{\vec{w}, b} \frac{\hat{\gamma}}{||\vec{w}||} \equiv \min_{\vec{w}, b} \frac{1}{2}||\vec{w}||^2$

$$
\min_{\vec{w}, b} \frac{1}{2}||\vec{w}||^2 \\

\ni y_i(\vec{w} \cdot \vec{x_i} + b) - 1 \geq 0
$$
This is a *Convex Quadratic Programming* Problem. Assume we have solved the $w^*$ and $b^*$ then we get the *maximum margin hyperplane* and *classification function*

That's the **Maximum Margin Method**

1. Construct and Solve the constrained optimization problem
    $$
    \min_{\vec{w}, b} \frac{1}{2}||\vec{w}||^2 \\

    \ni y_i(\vec{w} \cdot \vec{x_i} + b) - 1 \geq 0
    $$
    * Solve it and get the optimal solution $\vec{w}^*$ and $b^*$
2. Get the separating hyperplane and classification function
    * Separating hyperplane
        $$
        \vec{w}^*\cdot\vec{x} + b^* = 0
        $$
    * Classification function
        $$
            f(x) = \mathrm{sign}(\vec{w}^*\cdot\vec{x} + b^*)
        $$

---

### Support Vector

Support vectors are vectors that fulfill condition
$$
y_i(\vec{w}\cdot{\vec{x}_i + b) - 1 = 0}
$$

Consider two hyperplane when $y_i = 1$ and $y_i = -1$
$$
H_1: \vec{w}\cdot\vec{x} + b = 1 \\
H_2: \vec{w}\cdot\vec{x} + b = -1
$$

Distance between two hyperplane are called **margin**. Margin depends on normal vector $w$. Equal $\displaystyle\frac{2}{||w||}$

---

### Dual Algorithm

Apply *Lagrange Duality*, get the optimal solution of primal problem by solving dual problem

Advantage
1. Dual problem is easier to solve
2. Introduce kernel function, expend to non-linear classification problem

For each constraint introduce a **Lagrange multiplier** $\alpha_i \geq 0$

Lagrange function
$$
\mathcal{L}(\vec{w}, b, \vec{\alpha}) = \frac{1}{2} ||\vec{w}||^2 - \sum_{i=1}^N \alpha_iy_i (\vec{w}\cdot\vec{x}_i + b) + \sum_{i=1}^N \alpha_i
$$

According to Lagrange Duality. The dual problem of primal problem is a max-min problem.
$$
\max_{\vec{\alpha}} \min_{\vec{w}, b} \mathcal{L}(\vec{w}, b, \vec{\alpha})
$$

First solve $\displaystyle\min_{\vec{w}, b} \mathcal{L}(\vec{w}, b, \vec{\alpha})$ by find the derative respect to $\vec{w}$ and $b$ equal to zero.
$$
\nabla_{\vec{w}} \mathcal{L}(\vec{w}, b, \vec{\alpha}) = \vec{w} - \sum_{i=1}^N \alpha_i y_i \vec{x_i} = 0 \\
\nabla_b \mathcal{L}(\vec{w}, b, \vec{\alpha}) = - \sum_{i=1}^N \alpha_i y_i = 0 \\
$$

then we get

$$
\vec{w} = \sum_{i=1}^N \alpha_i y_i \vec{x_i} \\
\sum_{i=1}^N \alpha_i y_i = 0
$$

Substitute back to $\mathcal{L}(\vec{w}, b, \vec{\alpha})$, and get

$$
\min_{\vec{w}, b} \mathcal{L}(\vec{w}, b, \vec{\alpha}) = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\vec{x}_i \cdot \vec{x}_j) + \sum_{i=1}^N \alpha_i
$$

Second solve $\displaystyle\max_{\vec{\alpha}}$ problem (i.e. Dual problem)
$$
\max_{\vec{\alpha}} -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\vec{x}_i \cdot \vec{x}_j) + \sum_{i=1}^N \alpha_i \\
\ni \sum_{i=1}^N \alpha_i y_i = 0 \\
\forall \alpha_i \geq 0
$$

Change sign

$$
\min_{\vec{\alpha}} \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (\vec{x}_i \cdot \vec{x}_j) - \sum_{i=1}^N \alpha_i
$$

Then solve this problem you can get $\alpha^*$, and $\exist j \ni \alpha_j^* > 0$ then we can get $\vec{w}^*, b^*$

$$
\vec{w}^* = \sum_{i=1}^N \alpha_i^* y_i \vec{x_i} \\
b^* = y_j - \sum_{i=1}^N \alpha_i^* y_i (\vec{x}_i \cdot \vec{x}_j)
$$

Then we can substitute back our primal problem and get the separating hyperplane and classification function

---

### Solving alphas by SMO (Sequential Minimal Optimization)

The deduction above said that we definitely can get alphas and use it to get w and b. But didn't mention how.

It isn't possible to use the traditional way (i.e. Gradient Descent) since the dimension is too high (the same as input numbers).

So here is a efficient optimization algorithm, SMO.

It takes the large optimization problem and breaks it into many small problem.

#### Platt's SMO Algorithm

* Find a set of alphas and b.
    1. Once we have a set of alphas we can easily compute our weights w
    2. And get the separating hyperplane
* SMO algorithm choose two alphas to optimize on each cycle. Once a sutiable pair of alphas is found, one is increased and one is decreased.
    * Suitable criteria
        * A pair mus meet is that both of the alphas have to be outside their margin boundary
        * The alphas aren't already clamped or bounded
* The reason that we have to change two alphas at the same time is because we need have a constraint $\displaystyle \sum \alpha_i * y^{(i)} = 0$

## Introduce slack variable (C parameter)

## Using kernel for more complex data

## Reference

* 李航 - 統計機器學習
* Machine Learning in Action