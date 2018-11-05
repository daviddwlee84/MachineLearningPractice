# THE MNIST DATABASE of handwritten digits

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/)

* csv
    * [The MNIST Dataset of handwritten digits](http://makeyourownneuralnetwork.blogspot.com/2015/03/the-mnist-dataset-of-handwitten-digits.html)
    * [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/)
* sklearn (load from OpenML - [mnist_784](https://www.openml.org/d/554))
    * [fetch_openml](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml)
        ```python
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        ```
    * [fetch_mldata](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_mldata.html#sklearn.datasets.fetch_mldata) (Deprecated since version 0.20: Will be removed in version 0.22)
        ```python
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        ```

### Data Set Information

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

### Abstract

-|-
-|-
Data Set Characteristics |Image
Attribute Characteristics|Integer (0~255)
Number of Attributes     |784 (28x28 pixel)
Number of Instances      |42000 (original 60000)
Associated Tasks         |Classification

### Source

* Yann LeCun, Courant Institute, NYU
* Corinna Cortes, Google Labs, New York
* Christopher J.C. Burges, Microsoft Research, Redmond

## Result

Measure the accuracy of the test subset (30% of instances)

I have normalized all the data to [-1, 1].

* If data (pixel value) > 100 (threshold) then I'll give 1. Otherwise, -1.

### Binary Classifier (is 0 or not quesiton)

I also set over the label greater than 1 to be 1. And 0 to be -1. (The MUST step)

Use the last 50 row as training data

Model                     |Kernel|Accuracy|Parameters
--------------------------|------|--------|----------------
Binary SVM From Scratch   |Linear|1.0     |C = 1, tol = 0.001

Use the last 5000 row as training data

Model                     |Kernel|Accuracy|Parameters
--------------------------|------|--------|----------------
Binary SVM From Scratch   |Linear|0.9713  |C = 1, tol = 0.001
Binary SVM From Scratch   |RBF   |0.8960  |C = 5, gamma = 0.05, tol = 0.001

Ps. The cost of calculating RBF kernel of all training sample is too high to take. I haven't realize why sklearn can calculate so fast.

### Multi-class Classifier

Use the last 500 row as training data
Use the last 5500~500 row as testing data

Model                     |Kernel|Accuracy|Parameters
--------------------------|------|--------|----------------
OVR SVM From Scratch      |Linear|0.7886  |C = 1, tol = 0.001
OVR SVM From Scratch      |Linear|0.7266  |C = 100, tol = 0.001
OVR SVM From Scratch      |RBF   |0.7888  |C = 1, gamma = 1.3, tol = 0.0001

## Example

* [SVM MNIST handwritten digit classification](https://plon.io/explore/svm-mnist-handwritten-digit/USpQjoNcO8QHlmG6T)
    * [Github](https://github.com/ksopyla/svm_mnist_digit_classification)
