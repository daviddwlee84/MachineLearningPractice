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
#      Date:           Wed Mar 20 05:10:44 PDT 2019
#    Job ID:           13093.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Loading Epinions dataset...
Before training:
aMRR of training data: 0.000526879391481371
aMRR of test data: 0.007889554958122592
Training...
iteration: 1
F(U, V) = -98103.17873907623
Train MRR = 0.0719323001741936

...

iteration: 25
F(U, V) = -98055.11108214714
Train MRR = 0.07228035385966677
After training:
aMRR of training data: 0.07228035385966677
aMRR of test data: 0.3986766855682135
Result of U, V
U: [[0.00155989 0.00127414 0.00961984 ... 0.00305017 0.00890412 0.0060989 ]
 [0.00224032 0.00549899 0.01023754 ... 0.00691958 0.0018796  0.00505685]
 [0.00288271 0.01006467 0.00077869 ... 0.00959074 0.003221   0.00491805]
 ...
 [0.0079638  0.00488612 0.00648623 ... 0.00880086 0.00116553 0.00380049]
 [0.00670147 0.00488165 0.00480931 ... 0.00563985 0.00257467 0.00946336]
 [0.00714756 0.00979123 0.00570752 ... 0.00739845 0.00416285 0.00110496]]
V: [[0.11363496 0.11124309 0.11414293 ... 0.11204767 0.11657839 0.11507456]
 [0.00613668 0.00790209 0.00694653 ... 0.00844796 0.00217287 0.00945566]
 [0.02317879 0.02374501 0.02411474 ... 0.02392432 0.02865079 0.02885162]
 ...
 [0.00558333 0.0046503  0.00852283 ... 0.00199888 0.00897331 0.00586509]
 [0.00717035 0.00835769 0.00716895 ... 0.00173346 0.00265555 0.00064727]
 [0.00567358 0.00507568 0.00159209 ... 0.0009284  0.00999425 0.00897609]]

########################################################################
# End of output for job 13093.c009
# Date: Wed Mar 20 05:15:18 PDT 2019
########################################################################
```

### Tensorflow version

Training 25 iteration takes about 37 mins.

```txt
########################################################################
#      Date:           Wed Mar 20 05:10:30 PDT 2019
#    Job ID:           13092.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Loading Epinions dataset...
Before training:
aMRR of training data: 0.0007424622496369974
aMRR of test data: 0.005645436728149598
Training...
iteration: 1
F(U, V) = -98102.95931576437
Train MRR = 0.06666336258010003

...

 V) = -97991.26089012947
Train MRR = 0.066774728962234
After training:
aMRR of training data: 0.066774728962234
aMRR of test data: 0.3918667145748352
Result of U, V
U: <tf.Variable 'UnreadVariable' shape=(4718, 10) dtype=float64, numpy=
array([[0.01150841, 0.01147654, 0.00854206, ..., 0.00930481, 0.0108297 ,
        0.01678598],
       [0.012272  , 0.01352604, 0.01164434, ..., 0.00948663, 0.00815555,
        0.00920424],
       [0.01422384, 0.01305076, 0.01112218, ..., 0.00861753, 0.01294775,
        0.00796593],
       ...,
       [0.01452063, 0.01642285, 0.0156035 , ..., 0.01645473, 0.01500542,
        0.00765743],
       [0.01190807, 0.01639082, 0.0094909 , ..., 0.00842618, 0.00775538,
        0.01455897],
       [0.0113781 , 0.00918099, 0.0103875 , ..., 0.01535058, 0.01361814,
        0.01383631]])>
V: <tf.Variable 'UnreadVariable' shape=(49288, 10) dtype=float64, numpy=
array([[0.11068651, 0.11601034, 0.11537113, ..., 0.11112565, 0.11415173,
        0.11827064],
       [0.00283228, 0.00668931, 0.00435349, ..., 0.00756189, 0.00044096,
        0.00954167],
       [0.02736295, 0.02879317, 0.02379332, ..., 0.02938637, 0.03044311,
        0.03052954],
       ...,
       [0.00636845, 0.00415055, 0.00559802, ..., 0.00804753, 0.002897  ,
        0.0098266 ],
       [0.00972474, 0.00298496, 0.00649578, ..., 0.00678303, 0.00752719,
        0.00214504],
       [0.00462537, 0.00940434, 0.00704437, ..., 0.00392019, 0.00408214,
        0.00647309]])>

########################################################################
# End of output for job 13092.c009
# Date: Wed Mar 20 05:47:56 PDT 2019
########################################################################
```
