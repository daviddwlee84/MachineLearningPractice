# Support Vector Machine

## Brief Description

In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Supervised Learning|Classification, Regression, Outliers Detection|Numerous

* Support Vector Machine is suited for extreme cases
* SVM can be used to do *binary classification*
* SVM find a hyper-plane that separates its training data in such a way that the distance between the hyper plane and the cloest points form each class is maximized
* Implies that only *support vector* are important whereas other trainning examples are ignorable

* SVM can only be used on data that is *linear separable* (i.e. a hyper-plane can be drawn between the two groups)

* Advantage
    * Effective in high dimensional spaces
    * Effective in dimensions >> samples
    * Use a subset of training points in the decision function => Memory efficient
    * Different kernel functions for various decision functions
        * It's possible to add kernel functions together to achieve even more complex hyperplanes
* Disadvantage
    * Poor performance when features >> samples
    * SVMs do not provide probability estimates

## Concepts

### Terminology

* Hyperplane: Seperate different classes
* Support Vector

* Linear Separable

* Kernal Function

### Non-linear SVM

* LSVM (Linear SVM) => to apply when the class are linearly separable

* If have a data set that is not linear separable

    => Transform data into high dimensional feature space (make data linearly separable by map it to higher dimension)
    * e.g. 1D to 2D: Use a polynomial function to get a parabola $f(x) = x^2$

> But it's computationally expensive

### Kernal Function

* Use a kernal trick to reduce the computational cost
    * Kernel Function: Transform a non-linear space into a linear space
    * Popular kernel types
        * Linear Kernel

            $K(x, y) = x \times y$

        * Polynomial Kernel

            $K(x, y) = (x \times y + 1)^d$

        * Radial Basis Function (RBF) Kernel

            $K(x, y) = e^{-\gamma ||x-y||^2}$

        * Sigmoid Kernel
        * ...

### Tune Parameter

* Choosing the correct kernel is a non-trivial task. And may depend on specific task at hand.
    * Need to tune the kernel parameters to get good performance from a classifier
    * Parameter tuning technique
        * K-Fold
        * Cross Validation

## Links

### Tutorial

* [Youtube - Support Vector Machine](https://youtu.be/Y6RRHw9uN9o)

### Scikit Learn

* [Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html#svm)

### Wikipedia

* [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)