# Decision Tree

## Brief Description

A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Supervised Learning|Classification, Regression|Operations Research, Data Mining

#### Types

* Classification Tree: the predicted outcome is the class to which the data belongs.

* Regression Tree: The predicted outcome can be considered a real number (e.g. the price of a house, or a patient's length of stay in a hospital).

#### Family of Desition Tree Learning Algorithms

**Difference**: The procedure to decide which question to ask and when.

* ID3 (Iterative Dichotomiser 3)
* C4.5 (successor of ID3)
* C5.0
* CART (Classification And Regression Tree)
* CHAID (CHi-squared Automatic Interaction Detector): Performs multi-level splits when computing classification trees.
* MARS: extends decision trees to handle numerical data better.
* Conditional Inference Trees: Statistics-based approach that uses non-parametric tests as splitting criteria, corrected for multiple testing to avoid overfitting. This approach results in unbiased predictor selection and does not require pruning.

## Concept

### Decision Tree Learning

* Node
    * Receive a list of rows as input.
        * Root node: Receive the entire training set.
        * Leaf node: No further question to ask
    * Ask a true false question about one of the features.
        * Response to question => Split (partition) the data into two subsets
* Goal: Unmix the labels as we proceed down.
    * Produce the purest possible distribution of the labels at each node.
    * Perfectly unmixed: The input to a node contains only a single type of label.
        => No uncertainty on the type of label.
* Trick to building an effective tree: To understand which question to ask and when.
    * What type of questions can we ask about the data?
    * How to decide which question to ask when?
* Quantify how much a question helps to unmix the labels.
    * Quantify the amount of uncertainty at a single node using a metric - Gini Impurity
    * Quantify how much a question reduces that uncertainty using a concept - Information Gain
* Select the best question to ask at each point. And given that question, we'll recursively build the tree on each of the new nodes. We'll continue dividing the data until there are no further questions to ask, at which point we'll add a leaf.

### Splitting Criterion (Metrics)

Algorithms for constructing decision trees usually work top-down, by choosing a variable at each step that best splits the set of items.[14] Different algorithms use different metrics for measuring "best". These generally measure the homogeneity of the target variable within the subsets. These metrics are applied to each candidate subset, and the resulting values are combined (e.g., averaged) to provide a measure of the quality of the split.

* ID3: Information Gain
* C4.5: (Information) Gain Ratio
* CART: Gini Impurity(Index)

#### Gini Impurity - CART

**Impurity**: Chance of being incorrect if you randomly assign a label to an example in the same set. (i.e. The Uncertainty)

[**Information Entropy**](https://en.wikipedia.org/wiki/Entropy_(information_theory)): the average rate at which information is produced by a stochastic source of data.

#### Information Gain - ID3, C4.5

* Goal: Find the best question to ask (i.e. reduces our uncertainty the most).
    => A number to describe how much a question helps to unmix the labels at a node.
* Information Gain = Starting Impurity - Average Impurity of Child Nodes
    * Average Impurity (Weighted average of child nodes' uncertainty)
        * Weighted Average: We care more about a large set with lower uncertainty than a small set with high.
* Procedure: Keep tracking the question that produces the most gain. => the best one to ask at this node.

**Gain Ratio**: The normalized information gain.

### Pruning - to solve "Overfitting"

Decision-tree learners can create over-complex trees that do not generalize well from the training data. (This is known as overfitting.) Mechanisms such as pruning are necessary to avoid this problem (with the exception of some algorithms such as the Conditional Inference approach, that does not require pruning).

[Decision Tree Purning](https://en.wikipedia.org/wiki/Pruning_(decision_trees))

## Detail of Specific Type of the Algotithm

* Evolution: ID3 -> C4.5 -> C5.0

### ID3

[Wiki](https://en.wikipedia.org/wiki/ID3_algorithm)

### C4.5

* An algorithm to generate a decision tree.
* An extension of ID3 Algorithm. (same author, Ross Quinlan)
* Often reffered to as a Statistical Classifier.
* "A landmark decision tree program that is probably the machine learning workhouse most widely used in practice to date."

[Wiki](https://en.wikipedia.org/wiki/C4.5_algorithm)

### CART

Classification and regression trees (CART) are a non-parametric decision tree learning technique that produces either classification or regression trees, depending on whether the dependent variable is categorical or numeric, respectively.

[Wiki](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)

#### Pseudocode

Base cases:

* All the samples in the list belong to the same class. When this happens, it simply creates a leaf node for the decision tree saying to choose that class.
* None of the features provide any information gain. In this case, C4.5 creates a decision node higher up the tree using the expected value of the class.
* Instance of previously-unseen class encountered. Again, C4.5 creates a decision node higher up the tree using the expected value.

1. Check for the above base cases.
2. For each attribute a, find the normalized information gain ratio from splitting on a.
3. Let a_best be the attribute with the highest normalized information gain.
4. Create a decision node that splits on a_best.
5. Recur on the sublists obtained by splitting on a_best, and add those nodes as children of node.

[Wiki](https://en.wikipedia.org/wiki/C4.5_algorithm)

## Links

### Tutorial

* [Google Youtube Video - Decision Tree Classifier](https://youtu.be/LDRbO9a6XPU)
    * [Jupyter Notebook](https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb)
    * [Python File](https://github.com/random-forests/tutorials/blob/master/decision_tree.py)

### Scikit Learn

* [Scikit Learn User Guide - Decision Tree](http://scikit-learn.org/stable/modules/tree.html)

### Wikipedia

* [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)