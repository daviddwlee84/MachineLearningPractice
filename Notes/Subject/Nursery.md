# Nursery

## Dataset

[Nursery Data Set](https://archive.ics.uci.edu/ml/datasets/nursery)

### Data Set Information

Nursery Database was derived from a hierarchical decision model originally developed to rank applications for nursery schools. It was used during several years in 1980's when there was excessive enrollment to these schools in Ljubljana, Slovenia, and the rejected applications frequently needed an objective explanation. The final decision depended on three subproblems: occupation of parents and child's nursery, family structure and financial standing, and social and health picture of the family. The model was developed within expert system shell for decision making DEX (M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990.).

The hierarchical model ranks nursery-school applications according to the following concept structure.

NURSERY Evaluation of applications for nursery schools
. EMPLOY Employment of parents and child's nursery
. . parents Parents' occupation
. . has_nurs Child's nursery
. STRUCT_FINAN Family structure and financial standings
. . STRUCTURE Family structure
. . . form Form of the family
. . . children Number of children
. . housing Housing conditions
. . finance Financial standing of the family
. SOC_HEALTH Social and health picture of the family
. . social Social conditions
. . health Health conditions

Input attributes are printed in lowercase. Besides the target concept (NURSERY) the model includes four intermediate concepts: EMPLOY, STRUCT_FINAN, STRUCTURE, SOC_HEALTH. Every concept is in the original model related to its lower level descendants by a set of examples (for these examples sets see [Web Link]).

The Nursery Database contains examples with the structural information removed, i.e., directly relates NURSERY to the eight input attributes: parents, has_nurs, form, children, housing, finance, social, health.

Because of known underlying concept structure, this database may be particularly useful for testing constructive induction and structure discovery methods.

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Categorical
Number of Attributes     |8
Number of Instances      |12960
Associated Tasks         |Classification

### Source

* Creator:

    Vladislav Rajkovic et al. (13 experts)

* Donor:

    Marko Bohanec (marko.bohanec '@' ijs.si)

    Blaz Zupan (blaz.zupan '@' ijs.si)

## Result

Measure the accuracy of the test subset (30% of instances)

Model                            |Accuracy
---------------------------------|--------
Gaussian Naive Bayes Scikit Learn|0.6391
