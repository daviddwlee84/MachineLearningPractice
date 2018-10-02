# Letter Recognition

## Dataset

[Letter Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/letter+recognition)

### Data Set Information

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet. The character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000. See the article cited above for more details.

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Integer
Number of Attributes     |16
Number of Instances      |20000
Associated Tasks         |Classification

### Source

* Creator:

    David J. Slate

    Odesta Corporation; 1890 Maple Ave; Suite 115; Evanston, IL 60201

* Donor:

    David J. Slate (dave '@' math.nwu.edu) (708) 491-3867

## Result

Measure the accuracy of the test subset (30% of instances)

Model           |k|Accuracy
----------------|-|--------
kNN From Scratch|3|0.9522
kNN Scikit Learn|3|0.9482

## Related Resources

### Github

* [stephanedhml/ML-letter-recognition](https://github.com/stephanedhml/ML-letter-recognition)
* [davidgasquez/letter-recognition](https://github.com/davidgasquez/letter-recognition)