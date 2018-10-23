# Frequent Itemset Mining

## Dataset

* [Frequent Itemset Mining Dataset Repository](http://fimi.ua.ac.be/data/)
* [Retail Market Basket Data Set (original)](http://fimi.ua.ac.be/data/retail.dat)
* [Retail Market Basket Data Set (csv and other small sample set)](https://github.com/jiteshjha/Frequent-item-set-mining/tree/master/datasets)
* [Description pdf](http://fimi.ua.ac.be/data/retail.pdf)

### Data Set Information

The following dataset was donated by Tom Brijs and contains the (anonymized) retail market basket data from an anonymous Belgian retail store.

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate, Time-Series
Attribute Characteristics|Integer
Number of Attributes     |- (anonymized item numbers)
Number of Instances      |88163
Associated Tasks         |Interestingness, Association Rules

### Source

The data are provided ’as is’. Basically, any use of the data is allowed as long as the proper acknowledgment is provided and a copy of the work is provided to Tom Brijs.

## Result

Model                    |Minimum Support|Build Tree Time|Mine Tree Time
-------------------------|---------------|---------------|--------------
FP-Growth From Scratch   |1000           |06:00.018      |06:04.676
FP-Growth From Scratch   |5000           |05:56.779      |06:06.447

### Output frequent itemset

> Minimum Support = 1000

```python
[['39', '48'], ['39', '38'], ['48', '38'], ['39', '48', '38'], ['39', '32'], ['48', '32'], ['38', '32'], ['39', '48', '32'], ['39', '38', '32'], ['48', '38', '32'], ['39', '48', '38', '32'], ['39', '41'], ['48', '41'], ['38', '41'], ['32', '41'], ['39','48', '41'], ['39', '38', '41'], ['48', '38', '41'], ['39', '48', '38', '41'], ['39', '32', '41'], ['48', '32', '41'], ['39', '48', '32', '41'], ['39', '65'], ['48', '65'], ['39', '48', '65'], ['48', '89'], ['39', '89'], ['48', '39', '89'], ['39', '225'], ['48', '225'], ['39', '48', '225'], ['38', '170'], ['39', '170'], ['48', '170'], ['38', '39', '170'], ['38', '48', '170'], ['39', '48', '170'], ['38', '39', '48', '170'], ['39', '237'], ['48', '237'], ['39', '48', '237'], ['38', '36'], ['39', '36'], ['48', '36'], ['38', '39', '36'], ['38', '48', '36'], ['39', '48', '36'], ['38', '39', '48', '36'], ['38', '110'], ['39', '110'], ['48', '110'], ['38', '39', '110'], ['38', '48', '110'], ['39', '48', '110'], ['38', '39', '48', '110'], ['39', '310'], ['48', '310'], ['39', '48', '310'], ['39', '101'], ['48', '101'], ['39', '475'], ['48', '475'], ['39', '48', '475'], ['39', '271'], ['48', '271'], ['48', '413'], ['39', '413'], ['39', '438'], ['48', '438'], ['39', '1327'], ['39', '147'], ['48', '147'], ['39','270'], ['39', '2238'], ['39', '79'], ['39', '255'], ['48', '255'], ['38', '286'], ['38', '37']]
```

> Minimum Support = 5000

```python
[['39', '48'], ['39', '38'], ['48', '38'], ['39', '48', '38'], ['39', '32'], ['48', '32'], ['39', '48', '32'], ['39', '41'], ['48', '41'], ['39', '48', '41']]
```

## Example

* [jiteshjha/Frequent-item-set-mining - Apriori](https://github.com/jiteshjha/Frequent-item-set-mining)
