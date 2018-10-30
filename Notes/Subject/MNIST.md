# THE MNIST DATABASE of handwritten digits

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/)

* csv
    * [The MNIST Dataset of handwritten digits](http://makeyourownneuralnetwork.blogspot.com/2015/03/the-mnist-dataset-of-handwitten-digits.html)
    * [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/)
* sklearn
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

Use the last 5000 row as training data

Measure the accuracy of the test subset (30% of instances)

Model                     |Accuracy|Training Time
--------------------------|--------|-------------
