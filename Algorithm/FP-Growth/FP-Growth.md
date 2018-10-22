# FP Growth

## Brief Description

FP stands for frequent pattern

In the first pass, the algorithm counts occurrence of items (attribute-value pairs) in the dataset, and stores them to 'header table'. In the second pass, it builds the FP-tree structure by inserting instances. Items in each instance have to be sorted by descending order of their frequency in the dataset, so that the tree can be processed quickly. Items in each instance that do not meet minimum coverage threshold are discarded. If many instances share most frequent items, FP-tree provides high compression close to tree root.

### Quick View

Category|Usage|Application Field
--------|-----|-----------------
Unsupervised Learning|Association Rule Learning|-

## Links

### Example

* [eriklindernoren/ML-From-Scratch - FP Growth](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/fp_growth.py)

### Wikipedia

* [Association rule learning # FP-growth algorithm](https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm)
