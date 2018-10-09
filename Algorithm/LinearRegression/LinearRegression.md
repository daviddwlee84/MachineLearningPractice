# Linear Regression

## Brief Description

In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Supervised Learning|Regression|Many...

### Types

* Linear Regression
* Locally Weighted Linear Regression (LWLR)
* Ridge Regression
* Stagewise Linear Regression

## Concepts

### General approach to regression

* Train
    * Find the regression weights
* Test
    * Measure the R2
    * Correlation of the predicted value and tata, measure the success of the models

### Find regression weights

* input data: $X$
* regression weights: $w$

* predicted value: $\hat{y} = X^T w$

#### How to find $w$? => Minimize the error

* error defination: difference between predicted y and the actual y

Squared error: $\sum_{i=1}^{m}(y_i - x_i^T w)^2$

Squared error in matrix notation: $(y-Xw)^T(y-Xw)$

=> Take the derivative of this with respect to w => $X^T(y-Xw)$

Set this to 0 and solve for w to get: $\hat{w} = (X^TX)^{-1}X^Ty$

## Links

### Tutorial

* [Youtube - Linear Regression](https://youtu.be/CtKeHnfK5uA)

### Scikit Learn

* [sklearn.linear_model.LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)

### Wikipedia

* [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)