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

#### Accuracy

#### Recall Ratio

#### Confusion Matrix

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