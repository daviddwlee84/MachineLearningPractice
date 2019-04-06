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

```sh
# Logistic Regression
cd Algorithm/LogisticRegression/LogisticRegression_Iris
# comparison
python3 LogisticRegression_Iris_sklearn.py
# From Scratch
python3 LogisticRegression_Iris_FromScratch.py

# SVM (comparison)
cd Algorithm/SVM/SVM_Iris
python3 SVM_Iris_Multiclass.py
```

Measure the accuracy of the test subset (30% of instances)

Model                                    |Accuracy (avg. of 5)|Multi-class mode|Parameter
-----------------------------------------|--------------------|----------------|---------
Logistic Regression Scikit Learn         |0.9777              |ovr (default)   |-
LinearSVC Scikit Learn                   |0.9777              |ovr (default)   |-
SVM From Scratch (Binary)                |1.0                 |-               |(with standardlized data)
Logistic Regression From Scratch (Binary)|1.0                 |-               |(no influence)
SVM From Scratch                         |0.9111 (0.8133)     |ovr             |(with standardlized data)
Logistic Regression From Scratch         |0.7111 (0.6311)     |ovr             |max_iter=100, eta=0.01, standardlize=False
Logistic Regression From Scratch         |0.8444 (0.6311)     |multinomial     |max_iter=100, eta=0.01, standardlize=False
Logistic Regression From Scratch         |0.6222 (0.7733)     |ovr             |max_iter=1000, eta=0.0001, standardlize=False
Logistic Regression From Scratch         |**0.9777 (0.9777)** |multinomial     |max_iter=1000, eta=0.0001, standardlize=False
Logistic Regression From Scratch         |0.8222 (0.7288)     |ovr             |max_iter=100, eta=0.01, standardlize=True
Logistic Regression From Scratch         |0.8666 (0.8666)     |multinomial     |max_iter=100, eta=0.01, standardlize=True
Logistic Regression From Scratch         |0.8444 (0.8000)     |ovr             |max_iter=1000, eta=0.0001, standardlize=True
Logistic Regression From Scratch         |0.8444 (0.8444)     |multinomial     |max_iter=1000, eta=0.0001, standardlize=True

max_iter=100, eta=0.01, standardlize=False

```txt
Accuracy of Binary (only y==0 & y==1) Logistic Regression is: 1.0
Accuracy of Multi-class Logistic Regression with OVR is: 0.7111111111111111
average of 5: 0.6311111111111111
Accuracy of Multi-class Logistic Regression with Multinomial is: 0.6888888888888889
average of 5: 0.631111111111111
```

max_iter=1000, eta=0.0001, standardlize=False

```txt
Accuracy of Binary (only y==0 & y==1) Logistic Regression is: 1.0
Accuracy of Multi-class Logistic Regression with OVR is: 0.6222222222222222
average of 5: 0.7733333333333334
Accuracy of Multi-class Logistic Regression with Multinomial is: 0.9777777777777777
average of 5: 0.9777777777777776
```

max_iter=100, eta=0.01, standardlize=True

```txt
Accuracy of Binary (only y==0 & y==1) Logistic Regression is: 1.0
Accuracy of Multi-class Logistic Regression with OVR is: 0.8222222222222222
average of 5: 0.7288888888888889
Accuracy of Multi-class Logistic Regression with Multinomial is: 0.8666666666666667
average of 5: 0.8666666666666668
```

max_iter=1000, eta=0.0001, standardlize=True

```txt
Accuracy of Binary (only y==0 & y==1) Logistic Regression is: 1.0
Accuracy of Multi-class Logistic Regression with OVR is: 0.8444444444444444
average of 5: 0.8
Accuracy of Multi-class Logistic Regression with Multinomial is: 0.8444444444444444
average of 5: 0.8444444444444444
```

> SVM
>
> ```txt
> Accuracy of Binary (only y==0 (as -1) & y==1) SVM is: 1.0
> Accuracy of Scikit Learn Multi-class SVM is: 0.9777777777777777
> Accuracy of Multi-class SVM with OVR is: 0.9111111111111111
> average of 5: 0.8133333333333332
> ```

### Conclusion

1. Using smaller learning rate (eta) will get much slower (0.01 vs. 0.0001)
2. The performance of Multinomial LR became very stable when the learning rate is small enough
3. Doing standarize on data will increase the stability of OVR classifier but will lower the performance of Multinomial one
4. Using Multinomial with parameter (max_iter=1000, eta=0.0001, standardlize=False) will get the best performance (same as sklearn)

## Links

### EM Example

[EM Clustering of the Iris Dataset](https://swish.swi-prolog.org/example/iris.swinb)

### Softmax Example

* [UFLDL Tutorial - Softmax Regression](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
  * [exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
    * [solution (siddharth-agrawal/Softmax-Regression)](https://github.com/siddharth-agrawal/Softmax-Regression)
* [**rasbt/python-machine-learning-book - softmax-regression.ipynb**](https://github.com/rasbt/python-machine-learning-book/blob/master/code/bonus/softmax-regression.ipynb)
* [karan6181/Softmax-Classifier](https://github.com/karan6181/Softmax-Classifier)
