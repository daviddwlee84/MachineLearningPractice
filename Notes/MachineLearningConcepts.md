# Machine Learning Concepts

* Table of content
    * [Data Preprocessing](#Data-Preprocessing)
        * [Training and Test Sets - Splitting Data](#Splitting-Data)
        * [Missing Value](#Missing-Value)
    * [Model Evaluation](#Model-Evaluation)
        * [Classification](#Classification)
        * [Regression](#Regression)
        * [Clustering](#Clustering)
    * [Fitting and Model Complexity](#Fitting-and-Model-Complexity)
        * [Overfitting](#Overfitting)
        * [Underfitting](#Underfitting)
        * [Generalization](#Generalization)
        * [Regularization](#Regularization)
    * [Reducing Loss](#Reducing-Loss)
        * [Learning Rate](#Learning-Rate)
        * [Gradient Descent](#Gradient-Descent)
    * [Other Learning Method](#Other-Learning-Method)
        * [Lazy Learning](#Lazy-Learning)
        * [Incremental Learning (Online Learning)](#Incremental-Learning-(Online-Learning))

## Data Preprocessing

### Splitting Data

### Missing Value

* [Scikit Learn - 4.4 Imputation of missing values](http://scikit-learn.org/stable/modules/impute.html#impute)

#### Options

* Use the feature’s mean value from all the available data.
* Fill in the unknown with a special value like -1.
* Ignore the instance.
* Use a mean value from similar items.
* Use another machine learning algorithm to predict the value.

### Label Encoding

* [Scikit Learn - 4.9 Transforming the prediction target (y)](http://scikit-learn.org/dev/modules/preprocessing_targets.html#preprocessing-targets)

## Model Evaluation

### Classification

#### Accuracy (Error Rate)

* The error rate = the number of misclassified instances / the total number of instances tested.
* Measuring errors this way hides how instances were misclssified.

#### Confusion Matrix

* With a confusion matrix you get a better understanding of the classification errors.
* If the off-diagonal elements are all zero, then you have a perfect classifier

* Construct a confusion matrix: a table about Actual labels vs. Predicted label

#### Precision, Recall Ratio

These metrics that are more useful than error rate when detection of one class is more important than another class.

Consider a two-class problem.
(Confusion matrix with different outcome labeled)

Actual \ Redicted   |+1                 |-1
:------------------:|:-----------------:|:-----------------:
**+1**              |True Positive (TP) |False Negative (FN)
**-1**              |False Positive (FP)|True Negative (TN)

* **Precision** = TP / (TP + FP)
    * Tells us the fraction of records that were positive from the group that the classifier predicted to be positive

* **Recall** = TP / (TP + FN)
    * Measures the fraction of positive examples the classifier got right.
    * Classifiers with a large recall dont have many positive examples classified incorectly.

Summary:

* You can easily construct a classifier that achieves a high measure of recall or precision but not both.
* If you predicted everything to be in the positive class, you'd have perfect recall but poor precision.

#### ROC curve

[Wiki - Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

ROC stands for Receiver Operating Characteristic

* The ROC curve shows how the two rates chnge as the threshold changes
* The ROC curve has two lines, a solid one and a dashed one.
    * The solid line:
        * the leftmost point corresponds to classifying everything as the negative class.
        * the rightmost point corresponds to classifying everything in the positive class.
    * The dashed line:
        * the curve you'd get by randomly guessing.
* The ROC curve can be used to compare classifiers and make cost-versus-benefit decisions.
    * Different classifiers may perform better for different threshold values
* The best classifier would be in upper left as much as possible.
    * This would mean that you had a high true positive rate for a low false positive rate.

**AUC** (Area Under the Curve): A metric to compare different ROC

* The AUC gives an average value of the classifier's performance and doesn't substitute for looking at the curve.
* A perfect classifier would have an AUC of 1.0, and random guessing will give you a 0.5.

### Regression

#### Mean Absolute Error (MAE)

#### Mean Squared Error (MSE)

#### Root Mean Squared Error (RMSE)

### Clustering

#### Within Groups Sum of Squares

* Elbow method

#### Mean Silhouette Coefficient of all samples

#### Calinski and Harabaz score

* Max score => the best number of groups (clustes)

## Fitting and Model Complexity

### Overfitting

### Underfitting

### Generalization

### Regularization

## Reducing Loss

### Learning Rate

### Gradient Descent

## Other Learning Method

### Lazy Learning

### Incremental Learning (Online Learning)