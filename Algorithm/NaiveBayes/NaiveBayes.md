# Naive Bayes

## Brief Description

Naive bayes are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of *conditional independence* between every pair of features given the value of the class variable.

> But in the real word, in our concept, most of the things are not conditional independence. e.g. context in NLP

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Supervised Learning|Classification|Bayes' Theorem|

### Family of Naive Bayes

**Difference**: The assumptions that make regarding the distribution of $P(x_i|y)$

* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Complement Naive Bayes
* Bernoulli Naive Bayes

## Concept

### Bayes Decision

Posterior prob. = (Likelihood * Prior prob.) / Evidence

### Bayes' Theorem

$$
P(Y|X) = \frac{P(X|Y)\times P(Y)}{P(X)}
$$

$$
P(Y|X) = \frac{\prod_{i=1}^d P(X_i|Y)\times P(Y)}{P(X)}
$$

* [Wiki - Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

### Real-world conditions

* We predict label by multiplying them. But if any of these [probability is 0](#Zero-Probability-=>-Smoothing), then we will get 0 when we multiply them. To lessen the impact of this, we'll initialize all of our occurence counts to 1, and initialize the denominators to 2. (for binary classifier)
* Another problem is **Underflow**: doing too many multiplications of small numbers. (In programming, multiply many small numbers will eventually rounds off to 0 which called **floating-point underflow**)
    * Solution 1: Take the natural logarithm of this product

#### Zero Probability => Smoothing

> original: $P(w_k|c_j) = \displaystyle\frac{n_k}{n}$

m-estimation: $P(w_k|c_j) = \displaystyle\frac{n_k + mp}{n + m}$

> additional m "virtual samples" distributed according to p

## Application

### Document Classification/Categorization

Smoothing using Laplace smoothing (for $mp = 1$ and $m$ = Vocabulary)

$$
P(w_k|c_j) = \frac{n_k + 1}{n + |\operatorname{Vocabulary}|}
$$

### Word Sense Disambiguation

## TODO

* Figure out why the log mode in predictOne function has lower accuracy when using + than using * as the origin mode. ([Line 66](NaiveBayes_Nursery/NaiveBayes_Nursery_sklearn.py))

## Links

## Tutorial

* [Youtube - Naive Bayes Classifier](https://youtu.be/CPqOCI0ahss)

## Scikit Learn

* [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)

## Wikipedia

* [Additive smoothing (Laplace smoothing)](https://en.wikipedia.org/wiki/Additive_smoothing)
* [Bayesian Machine Learning](http://fastml.com/bayesian-machine-learning/)
* [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
