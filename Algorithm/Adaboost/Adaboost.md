# AdaBoost

[Labor Relations Data Set](https://archive.ics.uci.edu/ml/datasets/Labor+Relations)

## Brief Description

AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. AdaBoost is sensitive to noisy data and outliers. In some problems it can be less susceptible to the overfitting problem than other learning algorithms. The individual learners can be weak, but as long as the performance of each one is slightly better than random guessing, the final model can be proven to converge to a strong learner.

Every learning algorithm tends to suit some problem types better than others, and typically has many different parameters and configurations to adjust before it achieves optimal performance on a dataset, AdaBoost (with decision trees as the weak learners) is often referred to as the best out-of-the-box classifier. When used with decision tree learning, information gathered at each stage of the AdaBoost algorithm about the relative 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder-to-classify examples.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Supervised Learning|Boosting|

## Terminology

## Concept

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
