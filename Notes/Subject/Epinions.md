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

Running on the Intel DevCloud

* lambda: 0.001
* gamma: 0.0001
* dimension: 10
* max_iters: 25

### Numpy version

Training 25 iteration takes about 5 mins.

```txt
########################################################################
#      Date:           Thu Mar 21 07:18:51 PDT 2019
#    Job ID:           14025.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Loading Epinions dataset...
Before training:
aMRR of training data: 0.0006353374112811509
aMRR of test data: 0.007956903050108976
Training...
iteration: 1
F(U, V) = -98103.22940533378
Train MRR = 0.07364745799731802

...

iteration: 25
F(U, V) = -98055.0216934238
Train MRR = 0.07309697973242817
After training:
aMRR of training data: 0.07309697973242817
aMRR of test data: 0.37597077720560873
Result of U, V
U: [[0.00954796 0.00760001 0.0073078  ... 0.00372799 0.00778116 0.00689704]
 [0.00462603 0.00924769 0.00094167 ... 0.00702355 0.00157442 0.00070853]
 [0.00230983 0.00179425 0.00245876 ... 0.00295428 0.00410958 0.00430942]
 ...
 [0.00559139 0.0027931  0.0085679  ... 0.00490288 0.00630986 0.0042454 ]
 [0.00285912 0.00441831 0.0100436  ... 0.00538844 0.00287515 0.00402216]
 [0.00130894 0.00545737 0.00950588 ... 0.00652227 0.00977827 0.00721011]]
V: [[1.19275875e-01 1.11478172e-01 1.18963101e-01 ... 1.20614450e-01
  1.19919259e-01 1.17482801e-01]
 [6.16325541e-05 2.33317498e-03 8.25022308e-03 ... 7.65726315e-03
  3.76646350e-03 6.13837899e-05]
 [3.06319072e-02 2.20946055e-02 2.65286166e-02 ... 3.10524904e-02
  2.50954101e-02 2.55203229e-02]
 ...
 [3.95300778e-03 5.29186807e-03 4.99532383e-03 ... 3.60294039e-04
  1.22348676e-03 1.46227724e-03]
 [7.34118326e-03 6.55840976e-03 8.79186516e-03 ... 5.19756550e-03
  2.96185616e-03 4.25761235e-03]
 [1.65891046e-03 6.89145257e-03 1.01304498e-03 ... 8.08761405e-03
  9.06146803e-04 7.95355245e-03]]

########################################################################
# End of output for job 14025.c009
# Date: Thu Mar 21 07:23:35 PDT 2019
########################################################################
```

### Tensorflow version

Training 25 iteration takes about 38 mins.

```txt
########################################################################
#      Date:           Thu Mar 21 07:44:37 PDT 2019
#    Job ID:           14031.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Loading Epinions dataset...
Before training:
aMRR of training data: 0.0008913055200207466
aMRR of test data: 0.007605414294276319
Training...
iteration: 1
F(U, V) = -98102.91311650304
Train MRR = 0.0661797133834035

...

iteration: 25
F(U, V) = -97990.89009423851
Train MRR = 0.06640501426260449
After training:
aMRR of training data: 0.06640501426260449
aMRR of test data: 0.41430294776954524
Result of U, V
U: <tf.Variable 'UnreadVariable' shape=(4718, 10) dtype=float64, numpy=
array([[0.01223949, 0.00978324, 0.00752158, ..., 0.01712011, 0.0156768 ,
        0.01105841],
       [0.01031791, 0.01409   , 0.01347068, ..., 0.01407767, 0.01066915,
        0.00759094],
       [0.01540936, 0.00812375, 0.0146988 , ..., 0.01527205, 0.01200796,
        0.00906025],
       ...,
       [0.01661688, 0.0110754 , 0.00842814, ..., 0.00746034, 0.01646308,
        0.01548363],
       [0.01211186, 0.01448211, 0.01041163, ..., 0.01556238, 0.01517861,
        0.01350889],
       [0.01255056, 0.01511028, 0.0168943 , ..., 0.01686121, 0.01462265,
        0.01381591]])>
V: <tf.Variable 'UnreadVariable' shape=(49288, 10) dtype=float64, numpy=
array([[0.11327882, 0.11855019, 0.11442296, ..., 0.11006745, 0.11130216,
        0.11884053],
       [0.00285981, 0.00764639, 0.00138467, ..., 0.00478894, 0.00361345,
        0.00668478],
       [0.02445129, 0.02383463, 0.02831849, ..., 0.02605486, 0.02435821,
        0.0299549 ],
       ...,
       [0.00988245, 0.00448641, 0.00194927, ..., 0.00988818, 0.00860856,
        0.00119548],
       [0.00981348, 0.00417281, 0.00544336, ..., 0.00296357, 0.00201808,
        0.00600122],
       [0.00990922, 0.00979616, 0.00193402, ..., 0.00985568, 0.00462408,
        0.00851861]])>

########################################################################
# End of output for job 14031.c009
# Date: Thu Mar 21 08:22:07 PDT 2019
########################################################################
```
