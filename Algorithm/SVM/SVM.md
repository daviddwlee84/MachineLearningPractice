# Support Vector Machine

* [SVM Derivation](SVMDerivation.md)
* [Lagrange Multipliers and Constrained Optimization](../../Notes/Math/Calculus/MultivariableDeratives.md#Lagrange-Multipliers-and-Constrained-Optimization)
* [Lagrange Duality](../../Notes/Math/Calculus/MultivariableDeratives.md#Lagrange-Duality)

## Brief Description

Support Vector Machines (SVM) are learning systems that use a hypothesis space of *linear functions* in a *high dimensional* feature space, trained with a learning algorithm from *optimisation theory* that implements a *learning bias* derived from statistical learning theory.

### Quick View

Category|Usage|Methematics|Application Field
--------|-----|-----------|-----------------
Supervised Learning|Classification (Main), Regression, Outliers Detection Clustering (Unsupervised)|Convex Optimization, Constrained Optimization, Lagrange Multipliers|Numerous

* Support Vector Machine is suited for extreme cases (little sample set)
* SVM find a hyper-plane that separates its training data in such a way that the distance between the hyper plane and the cloest points form each class is maximized
* Implies that only *support vector* are important whereas other trainning examples are ignorable

* SVM can only be used on data that is *linear separable* (i.e. a hyper-plane can be drawn between the two groups)
* By using kernel trick, everything will be linear seprable in higher dimension

* Important Notes:
    * SVM natively only handles *binary classification*
        * You have to make sure your data label is either -1 or 1 !! (or it can never find the alphas so as support vectors)
        * For better [converage](#Converage-Problem) you have to [normalize](../../Notes/MachineLearningConcepts.md#Normalization) your data

Advantage

* Effective in high dimensional spaces
* Effective in dimensions >> samples
* Use a subset of training points in the decision function => Memory efficient
* Different kernel functions for various decision functions
    * It's possible to add kernel functions together to achieve even more complex hyperplanes

Disadvantage

* Poor performance when features >> samples
* SVMs do not provide probability estimates

SVM vs. Perceptron|SVM|Perceptron / NN
------------------|---|----------
**Solving Problem**|Optimization|Iteration
**Optimal**|Global (∵ convex)|Local
**Non-linear Seprable**|Higher dimension|Stack multi-layer model
**Performance**|Better with prior knowledge|Skip feature engineering step

## Terminology

* **Hyperplane**: The Decision Boundary that seperates different classes
    * 2 Dimension: line
    * 3 Dimension: plane
    * 4 Dimension or up: hyperplane
* **Support Vector**: The vectors that helps us to find the best hyperplane
* **Margin**: The space between support vectors and hyperplane

> To find a **hyperplane** best splits the data,
> because it is as far as possible from the **support vectors**
> which is another way of saying we maximized the **margin**

* Parameter
    * **C Parameter** (penalty parameter): Allows you to decide how much you want to penalize misclassified points
        * Low C: Prioritize simplicity (soft margin: because we allowing the miss classification)
        * High C: Prioritize making few mistakes
    * **Gamma Parameter** (for RBF, ploynomial, sigmoid kernel)
        * Small Gamma: Less complexity
        * High Gamma: More complexity


* Multiclass SVM (Decision function shape)
    * **OVR**: One vs. Rest (For each class make a binary classifier answer is or isn't the class)
        * Pros: Fewer classifications
        * Cons: Classes may be imbalanced
    * **OVO**: One vs. One
        * Pros: Less sensitive to imbalance
        * Cons: More classifications


* Linear Separable
* Kernel Function
    * Linear
    * Radial Basis Function (RBF)
    * Polynomial
    * Sigmoid

## Concepts

### Maximize the Margin

> It's a constrained optimization problem

* To solve constrained optimization problem is to use **Lagrange Multipliers** technique

> Convex quadratic programming problem ===[Lagrange Multipliers]===> Dual problem

* SMO (Sequential Minimal Optimization)


### Consider a non-linear separable problem

* Linear Support Vector Machine (LSVM) => to apply when the classes are linearly separable

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

#### C Parameter

### Multiple Classes

* MSVM (Multiclass Support Vector Machine)

#### Decision Function Shape

* OVR
* OVO

### Maximize Margin

## Converage Problem

* The simplified implementation is not guaranteed to coverage to the global optimum for all datasets!

* But it should always converge, unless there are numerical problems.
    Make sure the data is properly scaled. It is a bad idea if different features have values in different orders of magnitude. You might want to normalize all features to the range [-1,+1], especially for problems with more than 100 features.

### Libsvm FAQ about Converge

**Q: The program keeps running (with output, i.e. many dots). What should I do?**

In theory libsvm guarantees to converge. Therefore, this means you are handling ill-conditioned situations (e.g. too large/small parameters) so numerical difficulties occur.

You may get better numerical stability by replacing

typedef float Qfloat;
in svm.cpp with
typedef double Qfloat;
That is, elements in the kernel cache are stored in double instead of single. However, this means fewer elements can be put in the kernel cache.

**Q: The training time is too long. What should I do?**

For large problems, please specify enough cache size (i.e., -m). Slow convergence may happen for some difficult cases (e.g. -c is large). You can try to use a looser stopping tolerance with -e. If that still doesn't work, you may train only a subset of the data. You can use the program subset.py in the directory "tools" to obtain a random subset.

If you have extremely large data and face this difficulty, please contact us. We will be happy to discuss possible solutions.

When using large -e, you may want to check if -h 0 (no shrinking) or -h 1 (shrinking) is faster. See a related question below.

## Vary Large Datasets

* Paper - [Core Vector Machines: Fast SVM Training on Very Large Data Sets](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.445.4342&rep=rep1&type=pdf)

## SVMs in NLP

### Word Sense Disambiguation

### Text Categorization (TC)

#### Document Representation

> Document => Represent as a set of features

Features:

* Bag of words (BOW): "each word occurring in a document" remove "stop words"

Feature value:

* Binary: appear or not
* Integer: # of occurrences
* TF-IDF value
* ...

## Links

* [**CS229 Simplified SMO Algorithm**](http://math.unt.edu/~hsp0009/smo.pdf)

### Tutorial

* [Youtube - Support Vector Machine](https://youtu.be/Y6RRHw9uN9o)
* [Siraj Raval - Support Vector Machine](https://youtu.be/g8D5YL6cOSE)
    * [Classifying Data Using a Support Vector Machine](https://github.com/llSourcell/Classifying_Data_Using_a_Support_Vector_Machine)
* [Youtube - Support Vector Machines: A Visual Explanation with Sample Ptyhon Code](https://youtu.be/N1vOgolbjSc)
    * [adashofdata/muffin-cupcake](https://github.com/adashofdata/muffin-cupcake)

### MOOC

* NTU Hsuan-Tien Lin - [Machine Learning Technique](https://www.youtube.com/playlist?list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2)
    * Linear SVM
        * Course Introduction
        * Large-Margin Separating Hyperplane
        * Standard Large-Margin Problem
        * Support Vector Machine
        * Reasons behind Large-Margin Hyperplane
    * Dual SVM
        * Motivation of Dual SVM
        * Largange Dual SVM
        * Solving Dual SVM
        * Messages behind Dual SVM
    * Kernel SVM
        * Kernel Trick
        * Polynomial Kernel
        * Gaussian Kernel
        * Comparison of Kernels
    * Soft-Margin SVM
        * Motivation and Primal
        * Dual Problem
        * Messages
        * Model Selection

* Stanford Andrew Ng - CS229
    * [Optimization Object](https://youtu.be/hCOIMkcsm_g)
    * Large Margin Intuition
    * Mathematics Behind Large Margin Classification
    * Kernels-I, Kernels-II
    * Using a SVM

### Library

* Libsvm
    * [LIBSVM FAQ](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html)
* Scikit Learn
    * [Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html#svm)

### Wikipedia

* [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)

### Github

* [eriklindernoren/ML-From-Scratch - Support Vector Machine](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/support_vector_machine.py)