# Maximum Entropy Model

Maximum Entropy Classifier / [Multinomial Logistic Regression - i.e. Softmax](../LogisticRegression/LogisticRegression.md#Multinomial---Softmax-Regression-(SMR)),

> Can be considered to a mother of other algorithms
>
> [Condiitonal Random Field](../CRF/CRF.md)

## Brief Description

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Supervised Learning|Classification|Entropy|Many

## Concept

### The MEM Model

#### Background

Consider a machine learning problem

* $x$ = $(x_1, x_2, \dots, s_m)$ is input feature vector
* $y \in \{1, 2, \dots, k\}$ => a k classes classification problem

Given k linear model for machine learning. Each has dimension of m.

$$
\phi = w_{i1}x_1 + w_{i2}x_2+\cdots + w_{im}x_m,~~~1\leq i \leq k
$$

Prediction "class" $\hat{y}$ is the maximum "score" for each linear model output.

$$
\hat{y} = \arg\max_{1\leq i \leq k} \phi_i(x)
$$

TBD





### Training the Model

* GIS Algorithm
* IIS Algorithm
* Gradient Descent
* [Quasi-Newton Method](https://en.wikipedia.org/wiki/Quasi-Newton_method) (擬牛頓法) - L-BFGS Algorithm

#### GIS Algorithm

> GIS stands for Generalized Iterative Scaling

#### IIS Algorithm

> IIS stands for Improved Iterative Scaling. Improved from [GIS](#GIS-Algorithm)

### Solving Overfitting

* Feature Select: throw out rare feature
* Feature Induction: pick useful feature (improves performance)
* Smoothing

### Feature Selection

### Feature Induction

### Smoothing

## Application

> MEM is a classification model. It's not impossible to solve the sequential labeling problem, just not so suitable.
> For example of POS tagging, a classifier maybe not considered the global meaning information.

### POS Tagging

## Resources

### Wikipedia

* [Principle of maximum entropy - Maximum entropy models](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy#Maximum_entropy_models)
* [Multinomial logistic regression (Maximum entropy classifier)](https://en.wikipedia.org/wiki/Maximum_entropy_classifier)
