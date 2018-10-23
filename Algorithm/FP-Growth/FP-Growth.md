# FP Growth

## Brief Description

FP stands for frequent pattern

In the first pass, the algorithm counts occurrence of items (attribute-value pairs) in the dataset, and stores them to 'header table'. In the second pass, it builds the FP-tree structure by inserting instances. Items in each instance have to be sorted by descending order of their frequency in the dataset, so that the tree can be processed quickly. Items in each instance that do not meet minimum coverage threshold are discarded. If many instances share most frequent items, FP-tree provides high compression close to tree root.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Unsupervised Learning|Association Rule Learning|Frequent Itemset Mining

### Compare with Apriori

* FP-Growth builds from Apriori but uses some different techniques to accomplish the same task
* The task is finding frequent itemsets or pairs, sets of things that commonly occur together
* FP-Growth sorts the dataset in a special structure called an FP-tree
* FP-Growth does a better job of finding frequent itemsets, but it doesn't find association rules

### Steps

The FP-growth algorithm scans the dataset only twice

1. Build the FP-tree (Encode a dataset): pass counts the frequency of occurrence of all the items
2. Mine frequent itemsets from the FP-tree
    1. Get conditional pattern bases form the FP-tree
    2. From the conditional pattern base, construct a conditional FP-tree
    3. Recursively repeat steps 1 and 2 on until the tree contains a single item

### Pros and Cons

* Advantages
    * Much faster than Apriori (only need to go through data twice)
    * FP-tree ordering set in decending order. Same prefix will save storage space
    * Don't need to generate a list of candidates
* Disadvantage
    * Second time of iteration will hold lots of temperate values. Consume many storages
    * More expensive to build a FP-tree

## Concepts

### FP-tree Data Structure

The FP-tree is used to store the frequency of occurance for sets of items

* Feature
    * FP-tree has links connecting similar items (the linked items can be thought of as a linked list)
    * An item can appear multiple times in the same tree (unlike the search tree)
* Sets
    * Sets are stored as paths in the tree.
    * Sets with similar items will share part of the tree. Only when they differ will the tree split
* Node
    * A node identifies a single item from the set and the number of times it occurred in this sequence
* Path
    * A path will tell you how many times a sequence occurred
* Node Links
    * The links between similar items will be used to rapidly find the location of similar items
* Support (Minimum threshold)
    * Below support we considered items infrequent
    * (If an item is infrequent, supersets containing that item will also be infrequent)

## Links

### Example

* [eriklindernoren/ML-From-Scratch - FP Growth](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/fp_growth.py)

### Wikipedia

* [Association rule learning # FP-growth algorithm](https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm)
