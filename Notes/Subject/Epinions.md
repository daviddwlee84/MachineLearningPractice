# Epinions

## Dataset

[Epinions](http://www.trustlet.org/epinions.html)

### Description

Epinions is a website where people can review products. Users can register for free and start writing subjective reviews about many different types of items (software, music, television show, hardware, office appliances, ...). A peculiar characteristics of Epinions is that users are paid according to how much a review is found useful (Income Share program).

Also because of that, the attempts to game the systems are many and, as a possible fix, a trust system was put in place. Users can add other users to their "Web of Trust", i.e. "reviewers whose reviews and ratings they have consistently found to be valuable" and their "Block list", i.e. "authors whose reviews they find consistently offensive, inaccurate, or in general not valuable" (see the explanation of Epinions Web of Trust as backupped by Archive.org).

### Detail

The dataset contains

* 49,290 users who rated a total of
* 139,738 different items at least once, writing
* 664,824 reviews and
* 487,181 issued trust statements.

Users and Items are represented by anonimized numeric identifiers.

## Result

### Numpy version

Training for 25 iterations

```txt
Loading Epinions dataset...
Before training:
aMRR of training data: 0.0006130719857191306
aMRR of test data: 0.004124385195596999
Training
iteration: 1
F(U, V) = -98103.22300390419
Train MRR = 0.07504612035237727

...

iteration: 25
F(U, V) = -98055.02379605452
Train MRR = 0.07565748982409089
After training
aMRR of training data: 0.07565748982409089
aMRR of test data: 0.41272892396375155
```
