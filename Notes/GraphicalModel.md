# (Probabilistic) Graphical Model

## Overview

* Graph Models represent families of probability distribution via graphs
  * [Directed GM](#directed-graph-model): [Bayesian Network](#dag-bayesian-network)
    * Consider local information
  * [Undirected GM](#undirected-graph-model): [Markov Random Field](#probabilistic-undirected-graph-model-aka-markov-random-field)
    * Consider global information
  * Combination GM: Chain graphs

* Typical Graphical Model
  * [Generative Model](MachineLearningBigPicture.md#Generative-Model): from Category to Data
    * e.g. Naive Bayes
  * [Discriminative Model](MachineLearningBigPicture.md#Discriminative-Model): from Data to Category

## Directed Graph Model

> [HMM](../HMM/HMM.md) is a Directed Graph Model

### DAG (Bayesian Network)

## Undirected Graph Model

> [CRF is a Undirected Graph Model](#Probabilistic-Undirected-Graph-Model-(aka.-Markov-Random-Field))

Undirected Graph Model Concept:

* Clique (團): the fully connected subgraph (subset of nodes where each pair connected)
* Maximal clique (最大團): a clique that cannot be extended by including one more adjacent vertex (no extra node can be added and remain a clique)

Score Problem

* Potential Function (勢函數): map clique to a positive real number (score)

### Probabilistic Undirected Graph Model (aka. Markov Random Field)

> CRF = Markov Random Field + conditions; C => condition distribution, RF => Markov Random Field (joint distribution)
