# Iris

## Dataset

[Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)

* [Scikit-learn - The Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
* [Wiki - Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)

### Data Set Information

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain.

This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ). The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa" where the errors are in the second and third features.

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Real
Number of Attributes     |4
Number of Instances      |150
Associated Tasks         |Classification (but we will also do Clustering here)

### Attribute Information

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
    * Iris Setosa
    * Iris Versicolour
    * Iris Virginica

## Clustering with EM Algorithm

## Multi-class Classification with Logistic Regression

Model                                    |Accuracy (avg. of 5)|Multi-class mode
-----------------------------------------|--------------------|----------------
Logistic Regression Scikit Learn         |0.9777              |ovr (default)
Logistic Regression From Scratch (Binary)|1.0                 |-
Logistic Regression From Scratch         |0.8222 (0.7244)     |ovr

TODO: Multinomial (softmax)

## Links

### EM Example

[EM Clustering of the Iris Dataset](https://swish.swi-prolog.org/example/iris.swinb)
