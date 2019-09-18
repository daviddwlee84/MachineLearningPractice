# Recommendation System

## Brief Description

Definition: A subclass of *information filtering system*

To predict the *rating* or *preference* that user would give to an item (e.g. movie) or social element (e.g. people) they had not yet considered.

### Formalization

* Mapping Function: $f: U \times I \rightarrow R$
* Input:
  * User Model ($U$)
  * Item ($I$)
* Calculate:
  * Relativity ($R$) - used to **sorting**

### Search vs. Recommendation

Search: fulfilling users' **active needs**

* user know what he want
* user know how to describe

Recommend: **mining** and fulfilling users' **potential needs**

* user don't know where to find
* user don't know how to describe

### Purpose and Success Criteria

* **Prediction** perspective
* **Interaction** perspective
* **Conversion** perspective

## Background

### Informations Overload

Power laws / long-tailed distribution in the statistical sense

### Personas 用戶畫像

> customized recommendation / personalization

## Traditional Approach

* Collaborative Filtering 協同過濾 - users' social environment
  * Item-based
  * User-based
* Content-based Recommendation
  * analyzes the nature of each item (characteristics of items)
* Knowledge-based Recommendation

## Collaborative Filtering

> The process of filtering for information using techniques involving collaboration among multiple agents, viewpoints, etc.

works by taking a data set of user's data and comparing it to the data of other users

The key idea behind CF is that similar users share the same interest and that similar items are liked by a user.

> Basic assumption: Shared common interests in the past would still prefer similar products/items in the future (Those who agreed in the past tend to agree again in the future.)

### Approach

**Item-based or user-based similarity?**

Compared the distance between items is known as item-based similarity.

Compare the distance between users is known as user-based similarity.

The choice depends on how many users you may have or how many items you may have.

(If you have a lot of users, then you'll probably want to go with item-based similarity)

#### Item-based collaborative filtering

measure the similarity between the items that target users rates/ interacts with and other items

#### User-based collaborative filtering

measure the similarity between target users and other users

### Method

#### Association Rule Learning

Evaluation

* Confidence of A to B
    $$
    \operatorname{confidence}(A\Rightarrow B) = P(B|A)
    $$
* Support of A to B
    $$
    \operatorname{support}(A\Rightarrow B) = P(A \cup B)
    $$

Example:

1. bread, milk
2. bread, diaper, beer, eggs
3. milk, diaper, beer, coke
4. bread, milk, diaper, beer
5. bread, milk, diaper, coke

e.g. {Milk, Diaper} => Beer

$$
\operatorname{support} = \frac{\sigma(\text{milk}, \text{diaper}, \text{beer})}{|D|} = \frac{2}{5}
$$

$$
\operatorname{confidence} = \frac{\sigma(\text{milk}, \text{diaper}, \text{beer})}{\sigma(\text{milk}, \text{diaper})} = \frac{2}{3}
$$

* [Apriori](../Algorithm/Apriori/Apriori.md)

#### Naive Bayes Based Collaborative Filtering

* [Naive Bayes](../Algorithm/NaiveBayes/NaiveBayes.md)

#### Matrix Factorization (Singular Value Decomposition)

*Approximate the full matrix* by observing only *the most important feature* those with **

* [Singular Value Decomposition](../Algorithm/SVD/SVD.md)

Latent Factor Model

Objective Funciton: minimize squared error

Probability Matrix Factorization (PMF)

* [Wiki - Matrix factorization (recommender systems)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

#### Factorization Machine (FM)

* [Paper - Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
* [libFM: Factorization Machine Library](http://www.libfm.org/)
  * [github](https://github.com/srendle/libfm)

### Defect of CF

* cold start
* data sparsity
* popularity bias

Solution => Content-based Recommendation

## Content-based Recommendation

## Evaluation

Evaluation Experiment

1. *Offline* experiments - based on *historical data*
   * prediction accuracy, coverage
2. *Laboratory* studies - Controlled experiments
   * e.g. questionaries (survey)
3. Test with *real users* - **A/B tests**
   * e.g. sales increase, click through rates

### Rating Prediction ([Regression Evaluation](MachineLearningConcepts.md#Regression))

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Normalized Mean Absolute Error (NMAE)

### Top-N Prediction ([Classification Evaluation](MachineLearningConcepts.md#Classification))

* Precision
* Recall
* Accuracy
* F1-score
* AUC (Area Under Curve)
  * ROC Curve (Receiver Operating Characteristic Curve) (Sensitive Curve)

### Others

* Average Precision (AP)
* Mean Average Precision (MAP)
* Precision@N
  * e.g. P@5, P@10, P@20
* HR@H (Hit Rate)
  * e.g. HR@1, HR@5, HR@10
* Cumulative Gain (CG)
* Discounted Cumulative Gain (DCG)
* normalized Discounted Cumulative Gain (nDCG)
  * Ideal DCG (IDCG)

#### Diversity

* Intra-List Similarity (ILS)

#### Novelty

#### Coverage

## Popular Dataset

* MovieLens
* Netflix
* Book-Crossing
* Jester Joke
* Epinions
* Yelp
* BibSonomy
* Foursquare
* Flixster

## Resources

### Project

* [facebookresearch/dlrm](https://github.com/facebookresearch/dlrm) - Deep Learning Recommendation Model for Personalization and Recommendation Systems

### Book

Recommender Systems - The Textbook

* Ch3 Model-Based Collaborative Filtering
* Ch4 Content-Based Recommender System
* Ch5 Knowledge-Based Recommender System

### Wikipedia

* [Recommendation System](https://en.wikipedia.org/wiki/Recommender_system)
* [Information overload](https://en.wikipedia.org/wiki/Information_overload)
* [Long tail](https://en.wikipedia.org/wiki/Long_tail)
* [Persona](https://en.wikipedia.org/wiki/Persona)
* [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)

### Article

* [A Glimpse into Deep Learning for Recommender Systems](https://medium.com/@libreai/a-glimpse-into-deep-learning-for-recommender-systems-d66ae0681775)
* Machine Learning for Recommender systems
  * [art 1 (algorithms, evaluation and cold start)](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)
  * [Part 2 (Deep Recommendation, Sequence Prediction, AutoML and Reinforcement Learning in Recommendation)](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-2-deep-recommendation-sequence-prediction-automl-f134bc79d66b)
* Introduction to Recommender System.
  * [Part 1 (Collaborative Filtering, Singular Value Decomposition)](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
  * [Part 2 (Neural Network Approach)](https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7?gi=a21e975a20d3)

Evaluation Metrics

* [Recommender Systems — It’s Not All About the Accuracy](https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff)
