# Recommendation System

## Traditional Approach

* Content-based Recommendation
    * analyzes the nature of each item
* Collaborative Filtering
    * Item-based
    * User-based

### Collaborative filtering

works by taking a data set of user's data and comparing it to the data of other users

The key idea behind CF is that similar users share the same interest and that similar items are liked by a user.

**Item-based or user-based similarity?**

Compared the distance between items is known as item-based similarity.

Compare the distance between users is known as user-based similarity.

The choice depends on how many users you may have or how many items you may have.

(If you have a lot of users, then you'll probably want to go with item-based similarity)

#### Item-based collaborative filtering

measure the similarity between the items that target users rates/ interacts with and other items

#### User-based collaborative filtering

measure the similarity between target users and other users

### Singular Value Decomposition

## Evaluation

## Links

### Wikipedia

* [Recommendation System](https://en.wikipedia.org/wiki/Recommender_system)
* [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
* [Matrix factorization (recommender systems)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

### Article

* [A Glimpse into Deep Learning for Recommender Systems](https://medium.com/@libreai/a-glimpse-into-deep-learning-for-recommender-systems-d66ae0681775)
* Machine Learning for Recommender systems
    * [art 1 (algorithms, evaluation and cold start)](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)
    * [Part 2 (Deep Recommendation, Sequence Prediction, AutoML and Reinforcement Learning in Recommendation)](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-2-deep-recommendation-sequence-prediction-automl-f134bc79d66b)
* Introduction to Recommender System.
    * [Part 1 (Collaborative Filtering, Singular Value Decomposition)](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
    * [Part 2 (Neural Network Approach)](https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7?gi=a21e975a20d3)
