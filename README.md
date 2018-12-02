# Machine Learning Practice

Some practices using statistical machine learning technique based on some dataset.

To see more detail or example about deep learning, you can checkout my [Deep Learning](https://github.com/daviddwlee84/DeepLearningPractice) repository.

## Environment

* Using Python 3

(most of the relative path links are according to the repository root)

### Dependencies

* `numpy`: For low-level math operations
* `pandas`: For data manipulation
* `sklearn` - [Scikit Learn](http://scikit-learn.org/): For evaluation metrics, some data preprocessing

For comparison purpose

* `sklearn`: For machine learning models
* [`cvxopt`](http://cvxopt.org/): For convex optimization problem (for SVM)
* [`gensim`](https://radimrehurek.com/gensim/index.html): Topic Modelling

NLP related

* `hmmlearn`: Hidden Markov Models in Python, with scikit-learn like API
* `jieba`: Chinese text segementation library
* `pyHanLP`: Chinese NLP library (Python API)
* `nltk`: Natural Language Toolkit

## Projects

Subject|Technique / Task|Dataset|Solution|Notes
-------|--------------|-------|--------|-----
Letter Recognition|kNN / Classification|[Letter Recognition Datasets](https://archive.ics.uci.edu/ml/datasets/letter+recognition) ([File](Datasets/letter-recognition.csv))|[kNN From Scratch](Algorithm/kNN/kNN_Letter_Recognition/kNN_Letter_Recognition_FromScratch.py), [kNN Scikit Learn](Algorithm/kNN/kNN_Letter_Recognition/kNN_Letter_Recognition_sklearn.py)|[Notes](Notes/Subject/Letter_Recognition.md)
Page Blocks Classification|Decision Tree / Classification|[Page Blocks Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification) ([File](Datasets/page-blocks.csv))|[Decision Tree (CART) From Scratch](Algorithm/DecisionTree/DecisionTree_Page_Blocks_Classification/DecisionTree_Page_Blocks_Classification_FromScratch.py), [Decision Tree Scikit Learn](Algorithm/DecisionTree/DecisionTree_Page_Blocks_Classification/DecisionTree_Page_Blocks_Classification_sklearn.py)|[Notes](Notes/Subject/Page_Blocks_Classification.md)
CSM|Linear Regression / Regression|[CSM Dataset (2014 and 2015)](https://archive.ics.uci.edu/ml/datasets/CSM+%28Conventional+and+Social+Media+Movies%29+Dataset+2014+and+2015) ([File](Datasets/2014-and-2015-CSM-dataset.csv))|[Linear Regression From Scratch](Algorithm/LinearRegression/LinearRegression_CSM/LinearRegression_CSM_FromScratch.py), [Linear Regression Scikit Learn](Algorithm/LinearRegression/LinearRegression_CSM/LinearRegression_CSM_sklearn.py)|[Notes](Notes/Subject/CSM.md)
Nursery|Naive Bayes / Classification|[Nursery Data Set](https://archive.ics.uci.edu/ml/datasets/nursery) ([File](Datasets/nursery.csv))|[Gaussian Naive Bayes From Scratch](Algorithm/NaiveBayes/NaiveBayes_Nursery/NaiveBayes_Nursery_FromScratch.py), [Gaussian Naive Bayes Scikit Learn](Algorithm/NaiveBayes/NaiveBayes_Nursery/NaiveBayes_Nursery_sklearn.py)|[Notes](Notes/Subject/Nursery.md)
Post-Operative Patient|SVM (cvxopt) / Binary Classification|[Post-Operative Patient Data Set](http://archive.ics.uci.edu/ml/datasets/post-operative+patient) ([File](Datasets/post-operative.csv), [Simplified](Datasets/post-operative-binary.csv))|[SVM From Scratch (using cvxopt and simplified dataset)](Algorithm/SVM/SVM_Post_Operative_Patient/SVM_Post_Operative_Patient_Simplified_FromScratch.py), [SVM Scikit Learn](Algorithm/SVM/SVM_Post_Operative_Patient/SVM_Post_Operative_Patient_sklearn.py)|[Notes](Notes/Subject/Postoperative_Patient.md)
Student Performance|AdaBoost / Classification|[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance) ([File](Datasets/student-mat.csv))|[AdaBoost From Scratch](Algorithm/AdaBoost/AdaBoost_Student_Performance/AdaBoost_Student_Performance_FromScratch.py), [AdaBoost Scikit Learn](Algorithm/AdaBoost/AdaBoost_Student_Performance/AdaBoost_Student_Performance_sklearn.py)|[Notes](Notes/Subject/Student_Performance.md)
Sales Transactions|k-Means / Clustering|[Sales Transactions Dataset Weekly](http://archive.ics.uci.edu/ml/datasets/sales_transactions_dataset_weekly) ([File](Datasets/Sales_Transactions_Dataset_Weekly.csv))|[k-Means From Scratch](Algorithm/KMeans/KMeans_Sales_Transactions/KMeans_Sales_Transactions_FromScratch.py), [k-Means Scikit Learn](Algorithm/KMeans/KMeans_Sales_Transactions/KMeans_Sales_Transactions_sklearn.py)|[Notes](Notes/Subject/Sales_Transactions.md)
Frequent Itemset Mining|FP-Growth / Frequent Itemsets Mining|[Retail Market Basket Data Set](http://fimi.ua.ac.be/data/) ([File](Datasets/retail.csv))|[FP-Growth From Scratch](Algorithm/FP-Growth/FP-Growth_Frequent_Itemset_Mining/FP-Growth_Frequent_Itemset_Mining_FromScratch.py)|[Notes](Notes/Subject/Frequent_Itemset_Mining.md)
Automobile|PCA / Dimensionality Reduction|[Automobile Data Set](http://archive.ics.uci.edu/ml/datasets/Automobile) ([File](Datasets/imports-85.csv))|[PCA From Scratch](Algorithm/PCA/PCA_Automobile/PCA_Automobile_FromScratch.py), [PCA Scikit Learn](Algorithm/PCA/PCA_Automobile/PCA_Automobile_sklearn.py)|[Notes](Notes/Subject/Automobile.md)
Anonymous Microsoft Web Data|SVD / Recommendation System|[Anonymous Microsoft Web Data Data Set](https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data) ([File](Datasets/anonymous-msweb.csv), [Ratings Matrix](Datasets/MS_ratings_matrix.csv) [(by R)](Algorithm/SVD/SVD_Anonymous_Microsoft_Web_Data/Binary_Rating_Matrix.R))|[SVD From Scratch](Algorithm/SVD/SVD_Anonymous_Microsoft_Web_Data/SVD_Anonymous_Microsoft_Web_Data_FromScratch.py), [R Notebook - IBCF Recommender System](Algorithm/SVD/SVD_Anonymous_Microsoft_Web_Data/R_Notebook_IBCF_Recommender_System.Rmd)|[Notes](Notes/Subject/Anonymous_Microsoft_Web_Data.md)
Handwriting Digit|SVM (SMO) / Binary & Multi-class Classification|[MNIST](http://yann.lecun.com/exdb/mnist/) ([File](Datasets/MNIST.csv))|[Binary SVM From Scratch](Algorithm/SVM/SVM_MNIST/SVM_MNIST_Binary_FromScratch.py), [Multi-class (OVR) SVM From Scratch](Algorithm/SVM/SVM_MNIST/SVM_MNIST_Multiclass_FromScratch.py)|[Notes](Notes/Subject/MNIST.md)
Chinese text segmentation|HMM (EM)|[File](Datasets/Article/雅量.txt)|[HMM From Scratch](Algorithm/HMM/HMM_Text_Segmentation/HMM_FromScratch.py), [HMM hmmlearn](Algorithm/HMM/HMM_Text_Segmentation/HMMLearn.py), [Compare with Jieba and HanLP](Algorithm/HMM/HMM_Text_Segmentation/CompareJiebaHanLP.py)|-
Document Similarity and LSI|VSM, SVD|[Corpus of the People's Daily](http://dx.doi.org/10.18170/DVN/SEYRX5) ([File](Datasets/Article/199801_clear_1.txt))|[VSM From Scratch](Algorithm/VSM/VSM_Document_Similarity/VSM_Document_Similarity_FromScratch.py), [VSM Gensim](Algorithm/VSM/VSM_Document_Similarity/VSM_Document_Similarity_Gensim.py), [SVD/LSI Gensim](Algorithm/SVD/SVD_LSI/SVD_LSI_Gensim.py)|[Notes](Notes/Subject/Document_Similarity_LSI.md)

## Machine Learning Categories

### Consider the learning task

* **Surpervised Learning**
    * *Classification* - Discrete
    * *Regression* - Continuous
* **Unsupervised Learning**
    * *Clustering* - Discrete
    * *Dimensionality Reduction* - Continuous
    * *Association Rule Learning*
* **Semi-supervised Learning**
* **Reinforcement Learning**

### Cosider the desired output of a ML system

* *Classification*
    * [`Logistic Regression`](Algorithm/LogisticRegression/LogisticRegression.md) (optimization algo.)
    * [`k-Nearest Neighbors (kNN)`](Algorithm/kNN/kNN.md)
    * [`Support Vector Machine (SVM)`](Algorithm/SVM/SVM.md) - [Deduction](Algorithm/SVM/SVMDeduction.md) (optimization algo.)
    * [`Naive Bayes`](Algorithm/NaiveBayes/NaiveBayes.md)
    * [`Decision Tree (ID3, C4.5, CART)`](Algorithm/DecisionTree/DecisionTree.md)
* *Regression*
    * [`Linear Regression`](Algorithm/LinearRegression/LinearRegression.md) (optimization algo.)
    * [`Tree (CART)`](Algorithm/DecisionTree/DecisionTree.md)
* *Clustering*
    * [`k-Means`](Algorithm/KMeans/KMeans.md)
    * `Hierarchical Clustering`
* *Association Rule Learning*
    * [`Apriori`](Algorithm/Apriori/Apriori.md)
    * `Eclat`
    * [`FP-growth`](Algorithm/FP-Growth/FP-Growth.md) - Frequent itemsets mining
* *Dimensionality Reduction*
    * [`Principal Compnent Analysis (PCA)`](Algorithm/PCA/PCA.md)
    * [`Single Value Decomposition (SVD)`](Algorithm/SVD/SVD.md) - LSA, LSI, Recommendation System
    * `Linear Discriminant Analysis (LDA)`

### Ensemble Method (Meta-algorithm)

* Bagging
    * `Random Forests`
* Boosting
    * [`AdaBoost`](Algorithm/AdaBoost/AdaBoost.md) <- With some basic boosting notes
    * [`Gradient Boosting`](Algorithm/GradientBoosting/GradientBoosting.md)
        * `Gradient Boosting Decision Tree (GBDT)` (aka. Multiple Additive Regression Tree (MART))
    * [`XGBoost`](Algorithm/XGBoost/XGBoost.md)

### Others

* [`Hidden Markov Model (HMM)`](Algorithm/HMM/HMM.md)
* `Probabilistic Latent Semantic Analysis (PLSA)`
* `Latent Dirichlet Allocation (LDA)`
* [`Vector Space Model (VSM)`](Algorithm/VSM/VSM.md)

#### Heuristic Algorithm

* [SMO](Algorithm/SVM/SVMDeduction.md#Platt's-SMO-Algorithm) --> SVM
* [EM](Algorithm/HMM/HMM.md#EM-Algorithm) --> HMM

## [Machine Learning Concepts](Notes/MachineLearningConcepts.md)

### General Case

* [Data Preprocessing](Notes/MachineLearningConcepts.md#Data-Preprocessing)
    * [Normalization](Notes/MachineLearningConcepts.md#Normalization)
    * [Training and Test Sets - Splitting Data](Notes/MachineLearningConcepts.md#Splitting-Data)
    * [Missing Value](Notes/MachineLearningConcepts.md#Missing-Value)
    * [Dimensionality Reduction](Notes/MachineLearningConcepts.md#Dimensionality-Reduction)
* [Model Expansion](Notes/MachineLearningConcepts.md#Model-Expansion)
    * [Binary to Multi-class](Notes/MachineLearningConcepts.md#Binary-to-Multi-class)
* [Fitting and Model Complexity](Notes/MachineLearningConcepts.md#Fitting-and-Model-Complexity)
    * [Overfitting](Notes/MachineLearningConcepts.md#Overfitting)
    * [Underfitting](Notes/MachineLearningConcepts.md#Underfitting)
    * [Generalization](Notes/MachineLearningConcepts.md#Generalization)
    * [Regularization](Notes/MachineLearningConcepts.md#Regularization)
* [Reducing Loss](Notes/MachineLearningConcepts.md#Reducing-Loss)
    * [Learning Rate](Notes/MachineLearningConcepts.md#Learning-Rate)
    * [Gradient Descent](Notes/MachineLearningConcepts.md#Gradient-Descent)
* [Other Learning Method](Notes/MachineLearningConcepts.md#Other-Learning-Method)
    * [Cost-sensitive Learning](Notes/MachineLearningConcepts#Cost-sensitive-Learning)
    * [Lazy Learning](Notes/MachineLearningConcepts.md#Lazy-Learning)
    * [Incremental Learning (Online Learning)](Notes/MachineLearningConcepts.md#Incremental-Learning-(Online-Learning))
    * [Multi-label Classification](Notes/MachineLearningConcepts#Multi-label-Classification)

### Categorized

* Classification
    * Data Preprocessing
        * [Label Encoding](Notes/MachineLearningConcepts.md#Label-Encoding)
    * Real-world Problem
        * [Cost-sensitive Learning](Notes/MachineLearningConcepts.md#Cost-sensitive-Learning)
        * [Classification Imbalance](Notes/MachineLearningConcepts.md#Classification-Imbalance)
    * Evaluation Metrics
        * [Classification Metrics](Notes/MachineLearningConcepts.md#Classification)
    * [Binary to Multi-class Expension](Notes/MachineLearningConcepts.md#Binary-to-Multi-class)
* Regression
    * Evaluation Metrics
        * [Regression Metrics](Notes/MachineLearningConcepts.md#Regression)
* Clustering
    * Evaluation Metrics
        * [Clustering Metrics](Notes/MachineLearningConcepts.md#Clustering)

### Specific Field

* [Recommendation System](Notes/Recommendation_System.md)
    * Collaborative Filtering
* [Information Retrieval - Topic Modelling](Notes/Information_Retrieval.md)
    * Latent Semantic Analysis (LSA/LSI/SVD)
    * Latent Dirichlet Allocation (LDA)
    * Random Projections (RP)
    * Hierarchical Dirichlet Process (HDP)
    * word2vec

## Machine Learning Mathematics

### Topic

* Kernel Usages
* [Convex Optimization](Notes/Math/Topic/ConvexOptimization.md)

### Categories

* Linear Algebra
    * Orthogonality
    * Eigenvalues
    * Hessian Matrix
    * Quadratic Form
    * Markov Chain - HMM
* Calculus
    * [Multivariable Deratives](Notes/Math/Calculus/MultivariableDeratives.md)
        * Quadratic Approximations
        * Lagrange Multipliers and Constrained Optimization - SVM SMO
        * Lagrange Duality
* Probability and Statistics
    * Statistical Estimation
        * [Maximum Likelihood Estimation (MLE)](Notes/Math/Probability/MLE.md)

#### Basics

* Algebra
* Trigonometry

### Application

(from A to Z)

* Decision Tree
    * Entropy
* Naive Bayes
    * Bayes' Theorem
* PCA
    * Orthogonal Transformations
    * Eigenvalues
* SVD
    * Eigenvalues
* SVM
    * Convex Optimization
    * Constrained Optimization
    * Lagrange Multipliers
    * Kernel

## Books Recommendation

### Machine Learning

* [**Machine Learning in Action**](https://www.manning.com/books/machine-learning-in-action)
    * [Source Code](https://manning-content.s3.amazonaws.com/download/3/29c6e49-7df6-4909-ad1d-18640b3c8aa9/MLiA_SourceCode.zip)
* 統計學習方法 (李航)
* 機器學習 (周志華)

### Mathematics

* Linear Algebra with Applications (Steven Leon)
* Convex Optimization (Stephen Boyd & Lieven Vandenberghe)
* Numerical Linear Algebra (L. Trefethen & D. Bau III)

## Resources

### Tutorial

#### Videos

* [Google - Machine Learning Recipes with Josh Gordon](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)
    * [Josh Gordon's Repository](https://github.com/random-forests)
* [Youtube - Machine Learning Fun and Easy](https://www.youtube.com/playlist?list=PL_Nji0JOuXg2udXfS6nhK3CkIYLDtHNLp)
* [Siraj Raval - The Math of Intelligence](https://www.youtube.com/playlist?list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)
    * [Siraj Raval's Repository](https://github.com/llSourcell)
* [bilibili - 機器學習 - 白板推導系列](https://github.com/shuhuai007/Machine-Learning-Session)
* [bilibili - 機器學習升級版](https://www.bilibili.com/video/av22660033/)

#### Documentations

* [ApacheCN](http://ailearning.apachecn.org/) (ML, DL, NLP)
    * [Github - AiLearning](https://github.com/apachecn/AiLearning)
    * [Official Website - ApacheCN 中文開源組織](http://www.apachecn.org/)
* [Machine learning 101 (infographics)](http://usblogs.pwc.com/emerging-technology/machine-learning-101/)
    * [Machine learning overview (infographic)](http://usblogs.pwc.com/emerging-technology/a-look-at-machine-learning-infographic/)
    * [Machine learning methods (infographic)](http://usblogs.pwc.com/emerging-technology/machine-learning-methods-infographic/)
    * [Machine learning evolution (infographic)](http://usblogs.pwc.com/emerging-technology/machine-learning-evolution-infographic/)

#### Interactive Learning

* [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
* [Kaggle Learn Machine Learning](https://www.kaggle.com/learn/machine-learning)
* [Microsoft Professional Program - Artificial Intelligence track](https://academy.microsoft.com/en-us/professional-program/tracks/artificial-intelligence/)

#### MOOC

* [Stanford Andrew Ng - CS229](http://cs229.stanford.edu/)
    * [Coursera](https://www.coursera.org/learn/machine-learning)

### Github

* [Machine Learning from Scratch (eriklindernoren/ML-From-Scratch)](https://github.com/eriklindernoren/ML-From-Scratch)
* [Jack-Cherish/Machine-Learning](https://github.com/Jack-Cherish/Machine-Learning)

### Datasets

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html)
* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
* [資料集平台 Data Market](https://scidm.nchc.org.tw/)
* [AI Challenger Datasets](https://challenger.ai/datasets/)
* [Peking University Open Research Data](http://opendata.pku.edu.cn/)

### Machine Learning Platform

* [Kaggle](https://www.kaggle.com/)
* [OpenML](https://www.openml.org/)

### Machine Learning Tool

* [AutoML](https://www.automl.org/)
    * [Auto-sklearn](https://automl.github.io/auto-sklearn)
        * [github](https://github.com/automl/auto-sklearn)
