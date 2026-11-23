# Logistic Regression

## Brief Description

In statistics, the logistic model (or logit model) is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1". In the logistic model, the log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination of one or more independent variables ("predictors"); the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value).

### Quick View

| Category            | Usage          | Methematics               | Application Field |
| ------------------- | -------------- | ------------------------- | ----------------- |
| Supervised Learning | Classification | Gradient Descent, Sigmoid | Many...           |

## The Sigmoid function: a tractable step function

> (Heaviside) step function => can't be differential
>
> Sigmoid => differentiable

$$
\operatorname{sigmoid}(z) = \sigma(z) = \frac{1}{1+e^{-z}} = P(y=1)
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

## Multiple Classes

### Multinomial - Softmax Regression (SMR)

> Softmax Regression (synonyms: Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive)

Softmax

$$
\operatorname{softmax}(x)_i = \frac{ e^{x_i} }{ \sum_{j=1}^n e^{x_j} }
$$

### One-vs-All and One-vs-Rest

## Resources

* [Binary vs. Multi-Class Logistic Regression](https://chrisyeh96.github.io/2018/06/11/logistic-regression.html)
* [Google ML Crash Course - Multi-Class Neural Networks: Softmax](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax)

### Book

Dive into Deep Learning

* [Ch3.4. Softmax Regression](http://d2l.ai/chapter_linear-networks/softmax-regression.html)

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
* [Lecture 6.7 — Logistic Regression | MultiClass Classification OneVsAll — [Andrew Ng]](https://youtu.be/-EIfb6vFJzc)
* [Youtube - Softmax Regression (C2W3L08) — [Andrew Ng]](https://youtu.be/LLux1SW--oM)

### Wikipedia

* [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) - softmax regression

### Article

Binomial (sigmoid)

* [Logistic Regression from scratch in Python](https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac)

Multinomial (softmax)

* [2 Ways to Implement Multinomial Logistic Regression in Python](http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/) - use scikit learn
* [Machine Learning and Data Science: Multinomial (Multiclass) Logistic Regression](https://www.pugetsystems.com/labs/hpc/Machine-Learning-and-Data-Science-Multinomial-Multiclass-Logistic-Regression-1007/)
* [mlxtend - Softmax Regression](https://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/)
  * [jupyter notebook](https://github.com/rasbt/python-machine-learning-book/blob/master/code/bonus/softmax-regression.ipynb)
