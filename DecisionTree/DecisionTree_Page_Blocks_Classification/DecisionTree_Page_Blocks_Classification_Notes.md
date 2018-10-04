# Letter Recognition

## Dataset

[Page Blocks Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification)

### Data Set Information

The 5473 examples comes from 54 distinct documents. Each observation concerns one block. All attributes are numeric. Data are in a format readable by C4.5.

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Integer, Real
Number of Attributes     |10
Number of Instances      |5473
Associated Tasks         |Classification

### Source

* Original Owner:

    Donato Malerba
    Dipartimento di Informatica
    University of Bari

* Donor:

    Donato Malerba

## Result

Measure the accuracy of the test subset (30% of instances)

Model                     |Accuracy|Training Time
--------------------------|--------|-------------
Decision Tree Scikit Learn|0.9622  |00:00.035
Decision Tree From Scratch|0.9608  |05:58.906

## Can be improve

* Add support for missing (or unseen) attributes
* Prune the tree to prevent overfitting
* Add support for regression

## Related Resources

### Scikit Learn

* [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)