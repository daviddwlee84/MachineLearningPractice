# Post-Operative Patient Data Set

## Dataset

[Post-Operative Patient Data Set](http://archive.ics.uci.edu/ml/datasets/post-operative+patient)

### Data Set Information

The classification task of this database is to determine where patients in a postoperative recovery area should be sent to next. Because hypothermia is a significant concern after surgery (Woolery, L. et. al. 1991), the attributes correspond roughly to body temperature measurements.

Results:
-- LERS (LEM2): 48% accuracy

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Categorical, Integer
Number of Attributes     |8
Number of Instances      |90
Associated Tasks         |Classification
Missing Values?          |Yes

### Source

* Creators:

Sharon Summers, School of Nursing, University of Kansas
Medical Center, Kansas City, KS 66160
Linda Woolery, School of Nursing, University of Missouri,
Columbia, MO 65211

* Donor:

Jerzy W. Grzymala-Busse (jerzy '@' cs.ukans.edu) (913)864-4488

## Result

Paper - [Rule extraction from linear support vector machines](http://rexa.info/paper/77b535b98a279e3b1ee9499bead3408bc8d58c08)

Measure the accuracy of the test subset (30% of instances)

Model                          |Kernel      |Decision Function|Accuracy|Remark
-------------------------------|------------|-----------------|--------|------
SVM Scikit Learn               |RBF(default)|OVO & OVR        |0.7407  |use OVO and OVR are the same

Using simplified binary dataset (label I -> S)

Model                          |Kernel                   |Accuracy|Remark
-------------------------------|-------------------------|--------|------
SVM From Scratch (using cvxopt)|Linear & RBF & Polynomial|0.7407  |use Linear RBF and Polynomial kernel are the same
