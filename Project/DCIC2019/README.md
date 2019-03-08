# Digital China Innovation Contest 2019

## Competition

[Failure Prediction of Concrete Piston for Concrete Pump Vehicles](https://www.datafountain.cn/competitions/336/details)

### Description

Traditional approaches to production equipment maintenance remain mainly in corrective maintenance and routine-based maintenance. Corrective maintenance, which is no fail no repair, implies unscheduled downtime of equipment, severe security risk and unacceptable economic loss, while routine or time-based maintenance is time consuming and costly. By data analysis and modeling, predictive maintenance forecasts the residual life or possible failure of key components, and schedules maintenance accordingly，in order to avoid unscheduled downtime and to save cost.

Concrete pistons are key and consumptive components of concrete pump vehicles. Failure of the piston will cause malfunction of the pump vehicle, which may even halt the whole construction site and bring unacceptable economic loss. Piston life is closely related to the specific operation conditions of the equipment. First, we upload floor data of pump vehicles through Industrial Internet of Things (IIoT) to the cloud platform. Second, we train appropriate models based on the accumulated data. And then we predict the possible failure of the concrete piston, and accordingly remind the operators to carry out necessary maintenance before construction, which is promising to avoid economic loss caused by unscheduled downtime.

### Data

Data samples are extracted from floor data related to the concrete piston of some types of concrete pump vehicles, including worktime, engine rotation speed, hydraulic oil temperature, hydraulic pressure, etc. Sample labels indicate whether or not the piston will malfunction in the future on the given workload (pump volume). Contestants are expected to combine big data analysis and machine learning/deep learning techniques to extract features, establish failure prediction models， and forecast whether the piston will malfunction or not.

#### Files

Training Data

* data_train.zip
* train_labels.zip

We have many files within `data_train` with random rows (about 100 rows). Each file has a label of 0/1.

* data_test.zip
* submit_example.csv

#### Features

There are 14 features in the name of Chinese.

活塞工作時長，發動機轉速，油泵轉速，泵送壓力，液壓油溫，流量檔位，分配壓力，排量電流，低壓開關，高壓開關，攪拌超壓信號，正泵，反泵，設備類型

(work time, engine speed, pump speed, pumping pressure, oil temperature, flow stall, distribution pressure, displacement current, low voltage switch, high voltage switch, overpressure signal, positive pump, negative pump, equipment type)

And,

* 活塞工作時長：
  * 新換活塞後，累積工作時長
  * Numeric feature
* 發動機轉速、油泵轉速、泵送壓力、液壓油溫、流量檔位、分配壓力、排量電流：
  * 均為泵車的對應工況值
  * Numeric feature
* 低壓開關、高壓開關、攪拌超壓信號、正泵、反泵：
  * 開關量
* 設備類型：該泵車的類型
  * Categorical feature

除了開關量以外，上述設備類型、工況數據的具體值都經過了一定的脫敏處理(Data Masking)，即不完全是實際測量值，但已考慮盡量不影響數據蘊含的關係等信息。

> Ps. label只能為1或0，1表示該樣本對應的活塞在未來2000方工作量內，會出現故障；0標識在未來2000方工作量內不會故障。

### Evaluation

Macro-F1-Score

* [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  * macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

## Feature Engineering

### Cumulated Turning Rounds of Engine

累積工作時長 × 發動機轉速 (TODO)

## XGBoost Result

```sh
python3 xgboost_Kfold_f1score.py
```

### version 1 (baseline)

Basic feature engineering (for each uncategorical feature)

* unique length
* max
* min
* sum
* mean
* std

Result

* Local Score: 0.5991744450371395
  * [0.5939162417315951, 0.6012755671484955, 0.6033511178275852, 0.6088843119346735, 0.588444986543348]
* Online Score: 0.59446460000

## TODO

* [ ] Matplotlib Chinese font /or/ Transfer to English label
  * [matplotlib - Configuring the font family](https://matplotlib.org/gallery/api/font_family_rc_sgskip.html)
  * [How to Plot Unicode Characters with Matplotlib](https://jdhao.github.io/2018/04/08/matplotlib-unicode-character/)

## Links

* [**週冠軍分享|2019 DCIC 《混凝土泵車砼活塞故障預警》**](https://mp.weixin.qq.com/s?__biz=MzI5ODQxMTk5MQ==&mid=2247485857&idx=2&sn=148ad7cb473f75bf6719c46f1f75dfd6&chksm=eca77b19dbd0f20fb21cb6959383b6bbe942f256f5a03e5ed870ad463beeb91e247643d7874a&mpshare=1&scene=23&srcid=#rd)
* [**天線寶寶's baseline - jmxhhyx/DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles**](https://github.com/jmxhhyx/DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles)

### Packages

* [sklearn - Model persistence (joblib)](https://scikit-learn.org/stable/modules/model_persistence.html)
* [tqdm](https://github.com/tqdm/tqdm) - A fast, extensible progress bar for Python and CLI
