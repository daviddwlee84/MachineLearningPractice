# Multivariable Deratives

## Quadratic Approximations

## Lagrange Multipliers and Constrained Optimization

### Constrained Optimization

Maximize a multi-variable function with a constraint

**Example**:

Maximize $z = f(x, y) = x^2y$

on the set $\underbrace{x^2 + y^2 = 1}_{\text{Unit circle}}$ (to be a Constraint, we will represent in g later on)

We can imagine a 2D circle project onto the 3D surface and looking for the highest point.

=> Instead, we can use contour map (lines). Find a tangent contour line intersect the circle.

### Lagrange multipliers, using tangency to solve constrained optimization

To find the tangent line, we use *Gradient* (the gradient vectors, pass through a contour line, are perpendicular to it)

$f(x, y) = c$

$\nabla f$

$g(x, y) = x^2 y^2$

$\nabla g$

Both the gradient of f and g are perpendicular to its contour lines, and direct in the same way (length are proportional) => $\lambda$

$$
\nabla f(x_m, y_m) = \lambda \nabla g(x_m, y_m)
$$

We call $\lambda$ Lagrange Multiplier, and this function a Lagrangian Function.

**Core idea**:

Set these gradients equal to each other => Represent when the contour line for one function is tangent to the contour line of another.

$$
\nabla g = \nabla (x^2 + y^2) =
\begin{bmatrix}
2x \\
2y
\end{bmatrix}
$$
$$
\nabla x = \nabla (x^2y) =
\begin{bmatrix}
2xy \\
x^2
\end{bmatrix}
$$
$$
\begin{bmatrix}
2xy \\
x^2
\end{bmatrix}
= \lambda
\begin{bmatrix}
2x \\
2y
\end{bmatrix}
$$

The matrix form is the same as

$$
\begin{cases}
2xy = \lambda 2x \\
x^2 = \lambda 2y
\end{cases}
$$

We still need the third equation to solve three unknown, that is our constraint that we've know the whole time.

$$
\begin{cases}
2xy = \lambda 2x \\
x^2 = \lambda 2y \\
x^2 + y^2 = 1
\end{cases}
$$

These three equations characterize our constrained optimization problem.
> Top two equations tell us what's necessary in order for our contour lines, the contour of f and the contour of g to be perfectly tangent with each other.
> The bottom one just tells us that we have to be on the unit circle

Solve this example

(Each time you're dividing by a variable, you're basically assuming that it's not equal to zero. And so we need to check if it really not equal to zero.)

$$
\lambda = y \\
y = \pm \sqrt{1/3} \\
x = \pm \sqrt{2/3}
$$

In this four potential points (x, y), We're going to plug these point in f and see which point is the best.

So $\max f(x, y) = x^2y = \frac{2}{3}\sqrt{\frac{1}{3}}$

### The Lagrangian $\mathcal{L}$

Package the equations up as a function (and unpackage it again when we solve its gradient)

$$
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda (g(x, y) - b)
$$

(b is a constant)

$$
\nabla \mathcal{L} = 0
$$

$$
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial x} \\
\frac{\partial \mathcal{L}}{\partial y} \\
\frac{\partial \mathcal{L}}{\partial \lambda}
\end{bmatrix} =
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}
$$

from matrix form to equation form (top two)

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{\partial \mathcal{f}}{\partial x} - \lambda \frac{\partial \mathcal{g}}{\partial x}
= 0
\Rightarrow \frac{\partial \mathcal{f}}{\partial x}
= \lambda \frac{\partial \mathcal{g}}{\partial x}
$$

$$
\frac{\partial \mathcal{L}}{\partial y}
= \frac{\partial \mathcal{f}}{\partial y} - \lambda \frac{\partial \mathcal{g}}{\partial y}
= 0
\Rightarrow \frac{\partial \mathcal{f}}{\partial y}
= \lambda \frac{\partial \mathcal{g}}{\partial y}
$$

These steps simply set one of the gradient vectors proportional to the other one.

And finally the partial derivative of the Lagrangian with respect to the Lagrange multiplier:

$$
\frac{\partial \mathcal{L}}{\partial \lambda}
= 0 - (g(x, y) - b) = 0
\Rightarrow g(x, y) = b
$$

#### Summary of Lagrangian

* Computer is good at computing the gradient of some function equal to zero.
* If you construct the Lagrangian and then compute its gradient, all you're reallly doing is repackaging it up only to unpackage it again.
* The reason is because that's how you solve unconstrained maximization problems.
* The whole point of the Lagrangian is that it turns our *Constrained Optimization Problem* involving f and g and lambda into an *Unconstrained Optimization Problem*

### Meaning of the Lagrange multiplier

The example so far haven't use the variable lambda (usually eliminate in calculation). But it's not just some dummy variable.

Consider a more general problem,

$$
f(x, y) = \dots = M^* \\
g(x, y) = \dots = b
$$

M* is the maximum of function f, b is a constant. (We can pretend f is revenue and g is budget)

$$
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda(g(x, y) - b)\\
\nabla \mathcal{L} = \vec{0}
$$

When you solved $\nabla \mathcal{L}$. You may get many possible answer $(x^*, y^*, \lambda^*)$ that make it equal zero.
And we are simply try $(x^*, y^*)$ back to $f(x, y)$ and find the maximized $M^*$. 
$$
M^* = f(x^*, y^*)
$$

> $\lambda$ carry information about how much we can increase f if we increase g

Lets consider b is a variable. And we rewrite $M^*$ to be a function of b (because it depends on it).

$$
M^*(b) = f(x^*(b), y^*(b))
$$

Magical fact:

$$
\lambda^* = \frac{dM^*(b)}{d b}
$$

#### Proof

Plug in $(x^*, y^*, \lambda^*)$ into Lagrangian. (rather than f)

$$
\mathcal{L}(x^*, y^*, \lambda^*) = \underbrace{f(x^*, y^*)}_{M^*} - \lambda^*\underbrace{(g(x^*, y^*) - b)}_{=0}
$$

$g(x^*, y^*) - b$ must be euqal to zero because $x^*$ and $y^*$ have to satisfy the constraint.

$$
\mathcal{L}(x^*(b), y^*(b), \lambda^*(b), b) = f(x^*(b), y^*(b)) - \lambda^*(g(x^*(b), y^*(b)) - b)
$$

Find the derivative respect to b. (use multivariable chain rule)

$$
\frac{d\mathcal{L}^*}{db} = \frac{\partial\mathcal{L}}{\partial x^*} \cdot \frac{\partial h^*}{db} + \frac{\partial\mathcal{L}}{\partial y^*} \cdot \frac{\partial y^*}{db} + \frac{\partial\mathcal{L}}{\partial \lambda^*} \cdot \frac{\partial \lambda^*}{db} + \frac{\partial\mathcal{L}}{\partial b} \cdot \frac{db}{db}
$$

By the definition of $x^*, y^*, \lambda^*$ that it happends when $\nabla\mathcal{L} = 0$ that means the gradient of each partial derivative equals to zero.

$$
\frac{d\mathcal{L}^*}{db} = 0 \cdot \frac{\partial h^*}{db} + 0 \cdot \frac{\partial y^*}{db} + 0 \cdot \frac{\partial \lambda^*}{db} + \frac{\partial\mathcal{L}}{\partial b} \cdot 1 = \frac{\partial\mathcal{L}}{\partial b}
$$
The derivative of Lagrangi an is equal to the partial derivative of Lagrangian respect to b. ($\frac{d\mathcal{L}^*}{db} =\frac{\partial\mathcal{L}}{\partial b}$)

That means the *single-variable derivative of L with respect to b* ends up being the same as the *partial derivative of L*. This L, where you're free to change all the variables that these should be the same.

$$
\frac{d\mathcal{L}^*}{db} =\frac{\partial\mathcal{L}}{\partial b} = \lambda^*(b)
$$

---

### Lagrange Multipliers with Two Constraints

$$
\nabla f = \lambda_1 \nabla g_1 + \lambda_2 \nabla g_2,\\
g_1(x, y, z) = 0, g_2(x, y, z) = 0
$$

## Lagrange Duality

TBD

### Theorem (relationship between original and dual problem)

Karush-Kuhn-Tucker (KKT) Conditions:
$$
\nabla_x \mathcal{L}(x^*, \alpha^*, \beta^*) = 0 \\
\nabla_\alpha \mathcal{L}(x^*, \alpha^*, \beta^*) = 0 \\
\nabla_\beta \mathcal{L}(x^*, \alpha^*, \beta^*) = 0
$$
$$
\alpha_i^* c_i(x^*) = 0 \\
c_i(x^*) \leq 0
$$
$$
\alpha_i^* \geq 0 \\
h_j(x^*) = 0
$$
$\forall i = 1, 2, \dots, k, \forall j = 1, 2, \dots, k$

## Links

* [Applications of multivariable derivatives](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives)
    * Tangent planes and local linearization
    * Quadratic approximations
    * Optimizing multivariable functions
    * Optimizing multivariable functions (articles)
    * Lagrange multipliers and constrained optimization
    * Constrained optimization (articles)

## Reference

* [Lagrange Multipliers and Constrained Optimization](#Lagrange-Multipliers-and-Constrained-Optimization)
    * [Khan Academy - Lagrange multipliers and constrained optimization](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/constrained-optimization-introduction)
    * Tomas Calculus Ch 14.8 Lagrange Multipliers
* [Lagrange Duality](#Lagrange-Duality)
    * 李航 - 統計學習方法 Appendix C
