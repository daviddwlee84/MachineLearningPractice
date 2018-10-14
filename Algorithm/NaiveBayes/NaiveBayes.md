# Naive Bayes

## Brief Description

Naive bayes are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Supervised Learning|Classification|

### Family of Naive Bayes

**Difference**: The assumptions that make regarding the distribution of $P(x_i|y)$

* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Complement Naive Bayes
* Bernoulli Naive Bayes

## Concept

### Real-world conditions

* We predict label by multiplying them. But if any of these probability is 0, then we will get 0 when we multiply them. To lessen the impact of this, we'll initialize all of our occurence counts to 1, and initialize the denominators to 2. (for binary classifier)
* Another problem is **Underflow**: doing too many multiplications of small numbers. (In programming, multiply many small numbers will eventually rounds off to 0)
    * Solution 1: Take the natural logarithm of this product

## TODO

* Figure out why the log mode in predictOne function has lower accuracy when using + than using * as the origin mode. ([Line 66](NaiveBayes_Nursery/NaiveBayes_Nursery_sklearn.py))

## Links

## Tutorial

* [Youtube - Naive Bayes Classifier](https://youtu.be/CPqOCI0ahss)

## Scikit Learn

* [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)

## Wikipedia

* [Bayesian Machine Learning](http://fastml.com/bayesian-machine-learning/)
* [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)