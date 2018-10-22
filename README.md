# Machine Learning Practice

Some practices using statistical machine learning technique based on some dataset.

## Environment

* Using Python 3

### Dependencies

* `numpy`: For low-level math operations
* `pandas`: For data manipulation
* `sklearn` - [Scikit Learn](http://scikit-learn.org/): For evaluation metrics

## Projects

Subject|Technique|Dataset|Solution|Notes
-------|---------|-------|--------|-----
Letter Recognition|[kNN](Algorithm/kNN/kNN.md)|[Letter Recognition Datasets](https://archive.ics.uci.edu/ml/datasets/letter+recognition) ([File](Datasets/letter-recognition.csv))|[kNN From Scratch](Algorithm/kNN/kNN_Letter_Recognition/kNN_Letter_Recognition_FromScratch.py), [kNN Scikit Learn](Algorithm/kNN/kNN_Letter_Recognition/kNN_Letter_Recognition_sklearn.py)|[Notes](Notes/Subject/Letter_Recognition.md)
Page Blocks Classification|[Decision Tree](Algorithm/DecisionTree/DecisionTree.md)|[Page Blocks Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification) ([File](Datasets/page-blocks.csv))|[Decision Tree (CART) From Scratch](Algorithm/DecisionTree/DecisionTree_Page_Blocks_Classification/DecisionTree_Page_Blocks_Classification_FromScratch.py), [Decision Tree Scikit Learn](Algorithm/DecisionTree/DecisionTree_Page_Blocks_Classification/DecisionTree_Page_Blocks_Classification_sklearn.py)|[Notes](Notes/Subject/Page_Blocks_Classification.md)
CSM|[Linear Regression](Algorithm/LinearRegression/LinearRegression.md)|[CSM Dataset (2014 and 2015)](https://archive.ics.uci.edu/ml/datasets/CSM+%28Conventional+and+Social+Media+Movies%29+Dataset+2014+and+2015) ([File](Datasets/2014-and-2015-CSM-dataset.csv))|[Linear Regression From Scratch](Algorithm/LinearRegression/LinearRegression_CSM/LinearRegression_CSM_FromScratch.py), [Linear Regression Scikit Learn](Algorithm/LinearRegression/LinearRegression_CSM/LinearRegression_CSM_sklearn.py)|[Notes](Notes/Subject/CSM.md)
Nursery|[Naive Bayes](Algorithm/NaiveBayes/NaiveBayes.md)|[Nursery Data Set](https://archive.ics.uci.edu/ml/datasets/nursery) ([File](Datasets/nursery.csv))|[Gaussian Naive Bayes From Scratch](Algorithm/NaiveBayes/NaiveBayes_Nursery/NaiveBayes_Nursery_FromScratch.py), [Gaussian Naive Bayes Scikit Learn](Algorithm/NaiveBayes/NaiveBayes_Nursery/NaiveBayes_Nursery_sklearn.py)|[Notes](Notes/Subject/Nursery.md)
Post-Operative Patient|[SVM](Algorithm/SVM/SVM.md)|[Post-Operative Patient Data Set](http://archive.ics.uci.edu/ml/datasets/post-operative+patient) ([File](Datasets/post-operative.csv))|[SVM From Scratch](Algorithm/SVM/SVM_Post_Operative_Patient/SVM_Post_Operative_Patient_FromScratch.py)[SVM Scikit Learn](Algorithm/SVM/SVM_Post_Operative_Patient/SVM_Post_Operative_Patient_sklearn.py)|[Notes](Notes/Subject/Postoperative_Patient.md)
Student Performance|[Adaboost](Algorithm/Adaboost/Adaboost.md)|[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance) ([File](Datasets/student-mat.csv)))|[Adaboost Scikit Learn](Algorithm/Adaboost/Adaboost_Student_Performance/Adaboost_Student_Performance_sklearn.py)|[Notes](Notes/Subject/Student_Performance.md)
Labor Relations|-|[Labor Relations Data Set](https://archive.ics.uci.edu/ml/datasets/Labor+Relations)||[Notes](Notes/Subject/Labor_Relations.md)

## Machine Learning Categories

### Consider the learning task

* **Surpervised Learning**
    * *Classification*
    * *Regression*
* **Unsupervised Learning**
    * *Clustering*
    * *Association Rule Learning*

* **Semi-supervised Learning**
* **Reinforcement Learning**

### Cosider the desired output of a ML system

* *Classification*
    * `Logistic Regression`
    * `k-Nearest Neighbors (kNN)`
    * `Support Vector Machine (SVM)`
    * `Naive Bayes`
    * `Decision Tree (ID3, C4.5, CART)`
* *Regression*
    * `Linear Regression`
    * `Tree (CART)`
* *Clustering*
    * `k-Means`
    * `Hierarchical Clustering`
* *Association Rule Learning*
    * `Apriori`
    * `Eclat`
    * `FP-growth`

### Ensemble Method (Meta-algorithm)

* Bagging
    * `Random Forests`
* Boosting
    * `AdaBoost`

### Additional Tools

* Dimensionality Reduction
    * `Principal Compnent Analysis (PCA)`
    * `Single Value Decomposition (SVD)`
    * `Linear Discriminant Analysis (LDA)`
*  Big Data
    * `MapReduce`

## [Machine Learning Concepts](Notes/MachineLearningConcepts.md)

* Training and Test Sets - Splitting Data
* Overfitting
* Underfitting
* Generalization
* Regularization
* Reducing Loss
    * Learning Rate
    * Gradient Descent
        * Convex Function
        * Jesen's Inequality
        * Maximum Likelihood Estimation
        * Least Square Method
* Missing Value

## Mathematics

### Core

* Linear Algebra
    * Hessian Matrix
    * Quadratic Form
* Calculus
* Probability and Statistics

### Basics

* Algebra
* Trigonometry

## Books Recommand

* [**Machine Learning in Action**](https://www.manning.com/books/machine-learning-in-action)
    * [Source Code](https://manning-content.s3.amazonaws.com/download/3/29c6e49-7df6-4909-ad1d-18640b3c8aa9/MLiA_SourceCode.zip)

## Resources

### Tutorial

#### Videos

* [Google - Machine Learning Recipes with Josh Gordon](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)
    * [Josh Gordon's Repository](https://github.com/random-forests)
* [Youtube - Machine Learning Fun and Easy](https://www.youtube.com/playlist?list=PL_Nji0JOuXg2udXfS6nhK3CkIYLDtHNLp)

* [Siraj Raval - The Math of Intelligence](https://www.youtube.com/playlist?list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)
    * [Siraj Raval's Repository](https://github.com/llSourcell)

#### Documentations

* [ApacheCN](http://ailearning.apachecn.org/) (ML, DL, NLP)
    * [Github - AiLearning](https://github.com/apachecn/AiLearning)
    * [Official Website - ApacheCN 中文開源組織](http://www.apachecn.org/)

#### Interactive Learning

* [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
* [Kaggle Learn Machine Learning](https://www.kaggle.com/learn/machine-learning)

#### MOOC

* [Stanford Andrew Ng - CS229](http://cs229.stanford.edu/)
    * [Coursera](https://www.coursera.org/learn/machine-learning)

### Github

* [Machine Learning from Scratch (eriklindernoren/ML-From-Scratch)](https://github.com/eriklindernoren/ML-From-Scratch)

### Datasets

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html)
* [The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)