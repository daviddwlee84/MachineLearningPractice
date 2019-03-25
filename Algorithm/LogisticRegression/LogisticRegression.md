# Logistic Regression

## Brief Description

In statistics, the logistic model (or logit model) is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1". In the logistic model, the log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination of one or more independent variables ("predictors"); the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value).

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Supervised Learning|Classification|Gradient Descent, Sigmoid|Many...

## The Sigmoid function: a tractable step function

> (Heaviside) step function => can't be differential
>
> Sigmoid => differentiable

$$
\operatorname{sigmoid}(z) = \sigma(z) = \frac{1}{1+e^{-z}}
$$

### Training with Stochastic Gradient Ascent

Pseudocode

```txt
Start with the weights all set to 1
For each piece of data in the dataset:
    Calculate the gradient of one piece of data
    Update the weights vector by alpha*gradient
    Return the weights vector
```

## Logistic Discrimination

> In *logistic discrimination*, we don't model the class-conditional densities, but rather their ratio. (Assume that the log likelihood ratio is linear)

## Multiple Classes - Softmax

## Resources

### Book

Machine Learning in Action

* Ch5 Logistic Regression
  * Ch5.2.1 Gradient Ascent
  * Ch5.2.4 Stochastic Gradient Ascent

Introduction to Machine Learning

* Ch10.7 Logistic Discrimination
  * Ch10.7.2 Multiple Classes

### Scikit Learn

* [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Tutorial

* [Youtube - Logistic Regression](https://youtu.be/7qJ7GksOXoA)

### Wikipedia

* [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)