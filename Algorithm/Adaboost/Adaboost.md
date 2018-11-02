# AdaBoost

## Brief Description

AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. AdaBoost is sensitive to noisy data and outliers. In some problems it can be less susceptible to the overfitting problem than other learning algorithms. The individual learners can be weak, but as long as the performance of each one is slightly better than random guessing, the final model can be proven to converge to a strong learner.

Every learning algorithm tends to suit some problem types better than others, and typically has many different parameters and configurations to adjust before it achieves optimal performance on a dataset, AdaBoost (with decision trees as the weak learners) is often referred to as the best out-of-the-box classifier. When used with decision tree learning, information gathered at each stage of the AdaBoost algorithm about the relative 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder-to-classify examples.

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Supervised Learning|Boosting||

## Terminology

* Weak classifier: the classifier does a better job that randomly guessing but not by much.

## Concept

### Boosting

* In boosting, the different classifiers are trained sequentially.
* Each new classifier is trained based on the performance of those already trained.
* Boosting makes new clssifers focus on data that was previously misclassified by previous classifiers.

### Weak Learner - Decision Stump

A decision stump is a simple decision tree that make decision on one feature only. (a tree with only one split)

Pseudo-code:

```
Set the minError to +inf
For every feature in the dataset:
    For every step:
        For each inequality:
            Build a decision stump and test it with the weighted dataset
            If the error is less than minError: set this stump as the best stump
Return the best stump
```

### AdaBoost Algorithm

Variables:

* weight vector: $D$
    * Initially are all equal
* error $\displaystyle\varepsilon = \frac{\textit{number of incorrectly classified examples}}{\textit{total number of examples}}$
* $\displaystyle\alpha=\frac{1}{2}\ln(\frac{1-\varepsilon}{\varepsilon})$
    * assign to each of the classifiers
    * are based on the error of each weak classifier

Update $D$ by calculated $\alpha$
$$
D_i^{(t+1)} =
\begin{cases}
    \frac{D_i^{(t)} e^{-\alpha}}{\mathit{Sum}(D)} \text{, if correctly predicted} \\
    \frac{D_i^{(t)} e^{\alpha}}{\mathit{Sum}(D)} \text{, if incorrectly predicted}
\end{cases}
$$

After $D$ is calculated, AdaBoost starts on the next iteration.

AdaBoost repeats the training and weight-adjusting iterations until

* the training error is 0
* the number of weak classifier reaches a user-defined value

Pseudo-code:

```
For each iteration:
    Find the best stump using buildStump()
    Add the best stump to the stump array
    Calculate alpha
    Calculate the new weight vector - D
    Update the aggregate class estimate
    If the error rate == 0.0: break out of the for loop
```

## Links

## MOOC

* NTU Hsuan-Tien Lin - Machine Learning Technique - Adaptive Boosting
    * [Motivation of Boosting](https://youtu.be/hL8DjIHAzZY)
    * [Diversity by Re-weighting](https://youtu.be/pTNKUj_1Dw8)
    * [Adaptive Boosting Algorithm](https://youtu.be/vqTXLTYqbbw)
    * [Adaptive Boosting in Action](https://youtu.be/5wPN87bwoaE)

## Scikit Learn

* [Ensemble Methods](http://scikit-learn.org/stable/modules/ensemble.html)

* [sklearn.ensemble.AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
* [sklearn.ensemble.AdaBoostRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor)

## Wikipedia

* [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)

## Github

* [eriklindernoren/ML-From-Scratch - AdaBoost](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/adaboost.py)
