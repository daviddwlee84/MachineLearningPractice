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
  * Binary feature
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

累積工作時長 × 發動機轉速 (tried in [version 2](#version-2))

累積工作時長 × 油泵轉速 (tried in version 3)

## XGBoost Result

```sh
python3 xgboost_Kfold_f1score.py
```

> Each version means more significant change.
> Each version may have multiple test assignment with a little difference.

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

### version 2

Feature added

* first derivative (without 低壓開關, 高壓開關, 攪拌超壓信號, 正泵, 反泵, because these value basically won't change)
  * [numpy.gradient](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html)
  * [pandas.DataFrame.diff](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.diff.html)
    * [Stackoverflow - python pandas: how to calculate derivative/gradient](https://stackoverflow.com/questions/41780489/python-pandas-how-to-calculate-derivative-gradient)
* [total engine turn](#Cumulated-Turning-Rounds-of-Engine)

Feature removed

* Remove min, max, for 活塞工作時長, 高壓開關, 低壓開關
* Remove unique length
* Remove data_drop_dup_len

Result (without `total_turn`)

```txt
Feature Importance: ( 31 )
 {'活塞工作时长_mean': 1, 'data_len': 6, '活塞工作时长_sum': 7, '高压开关_sum': 3, '排量电流_sum': 2, '油泵转速_1st_deri_std': 2, '液压油温_max': 1, '泵送压力_sum': 1, '液压油温_min': 4, '低压开关_sum': 2, '泵送压力_max': 3, '发动机转速_1st_deri_min': 1, '发动机转速_min': 1, '分配压力_max': 3, '排量电流_max': 1, '油泵转速_min': 1, '液压油温_mean': 1, '泵送压力_min': 1, '泵送压力_1st_deri_max': 2, '液压油温_1st_deri_min': 2, '流量档位_sum': 1, '排量电流_1st_deri_min': 1, '发动机转速_max': 1, '分配压力_min': 1, '泵送压力_1st_deri_sum': 3, '流量档位_mean': 1, '泵送压力_1st_deri_min': 1, '泵送压力_1st_deri_std': 1, '流量档位_max': 1, '发动机转速_mean': 1, '分配压力_mean': 1}
```

* Local Score: 0.5983421437352534
  * [0.5892376117944992, 0.6074827461848528, 0.6054303474690202, 0.6009050802877793, 0.5886549329401154]
* Online Score: 0.60010433

Result (with `total_engine_turn`)

```txt
Feature Importance: ( 32 )
 {'活塞工作时长_mean': 1, 'data_len': 7, '活塞工作时长_sum': 5, '高压开关_sum': 3, '排量电流_sum': 2, '油泵转速_1st_deri_std': 2, '液压油温_max': 1, '泵送压力_sum': 2, '液压油温_min': 3, '低压开关_sum': 2, '泵送压力_max': 3, '发动机转速_1st_deri_min': 1, '发动机转速_min': 2, '排量电流_max': 1, '油泵转速_min': 1, '排量电流_min': 1, '液压油温_1st_deri_sum': 1, '油泵转速_sum': 2, 'total_turn': 2, '分配压力_mean': 2, '排量电流_1st_deri_min': 1, '发动机转速_max': 1, '液压油温_1st_deri_min': 2, '分配压力_min': 1, '泵送压力_1st_deri_sum': 2, '泵送压力_1st_deri_max': 1, '流量档位_sum': 1, '泵送压力_1st_deri_min': 1, '反泵_sum': 1, '流量档位_max': 1, '油泵转速_1st_deri_max': 1, '发动机转速_mean': 1}
```

* Local Score: 0.6004209943823582
  * [0.5915075875237326, 0.6069075066539147, 0.6080411032386698, 0.609920879474779, 0.5857278950206952]
* Online Score: 0.59949291000

### version 3 (no improvement)

Feature added

* total pump turn

Feature modified

* Make clear of binary feature and other numeric feature

```txt
Feature Importance: ( 34 )
 {'活塞工作时长_mean': 1, 'data_len': 6, '活塞工作时长_sum': 5, '高压开关_sum': 3, '排量电流_sum': 2, '分配压力_mean': 2, '液压油温_max': 1, '泵送压力_mean': 1, '分配压力_sum': 1, '低压开关_sum': 2, '泵送压力_max': 3, 'pump_turn_mean': 2, '发动机转速_min': 1, '分配压力_max': 3, '排量电流_max': 1, 'pump_turn_1st_deri_min': 1, '发动机转速_max': 2, '泵送压力_min': 1, '泵送压力_std': 1, '液压油温_1st_deri_min': 1, '液压油温_min': 3, '流量档位_sum': 1, '排量电流_1st_deri_max': 1, '排量电流_1st_deri_std': 1, 'engine_turn_min': 1, '发动机转速_1st_deri_sum': 1, '泵送压力_1st_deri_max': 2, 'engine_turn_sum': 1, '流量档位_mean': 1, '泵送压力_1st_deri_min': 1, '泵送压力_1st_deri_std': 1, '流量档位_max': 2, '发动机转速_sum': 1, 'engine_turn_max': 1}
```

* Local Score: 0.6016200155116346
  * [0.595883290960082, 0.6060052612666093, 0.6087465431550466, 0.6089840073376056, 0.5884809748388296]
* Online Score: 0.59313458000

### version 4 (no improvement)

Feature added

* oil temperature abnormal
* engine too fast

Result:

```txt
Feature Importance: ( 38 )
 {'engine_turn_mean': 2, 'data_len': 5, '活塞工作时长_sum': 3, '高压开关_sum': 5, '排量电流_sum': 3, 'engine_turn_max': 1, '液压油温_max': 1, '泵送压力_mean': 2, '分配压力_sum': 2, 'engine_turn_min': 1, 'pump_turn_mean': 1, '发动机转速_min': 1, '分配压力_max': 2, 'pump_turn_1st_deri_min': 2, '分配压力_std': 1, '低压开关_sum': 2, '油泵转速_sum': 1, '液压油温_min': 2, 'pump_turn_1st_deri_max': 1, '发动机转速_sum': 2, 'pump_turn_sum': 1, '泵送压力_max': 1, 'pump_turn_1st_deri_std': 1, '排量电流_std': 1, '发动机转速_max': 1, '泵送压力_1st_deri_max': 1, '分配压力_mean': 1, 'engine_turn_1st_deri_max': 1, '流量档位_mean': 1, '流量档位_sum': 1, '流量档位_max': 1, '油泵转速_1st_deri_sum': 1, '排量电流_1st_deri_mean': 1, '油泵转速_mean': 1, 'pump_turn_1st_deri_sum': 1, '油泵转速_min': 1, '排量电流_min': 1, '泵送压力_min': 1}
```

* Local Score: 0.5986287502997885
  * [0.5947046535002127, 0.6027999523632018, 0.6096115150976127, 0.5936959028100426, 0.5923317277278726]
* Online Score: 0.58178872000

Some change but doesn't improve (see commit history for detail)

```txt
Feature Importance: ( 39 )
 {'发动机转速_1st_deri_fft_sum': 1, '泵送压力_sum': 2, '分配压力_fft_std': 1, '活塞工作时长_sum': 4, '分配压力_mean': 1, '分配压力_fftmax_min_sub': 1, '泵送压力_1st_deri_sum': 1, '油泵转速_fft_sum': 1, '液压油温_max': 1, '排量电流_1st_deri_sum': 1, '分配压力_1st_deri_fft_sum': 1, '泵送压力_fft_sum': 1, '泵送压力_max': 1, '液压油温_mean': 2, '高压开关_sum': 3, 'engine_turn_min': 2, 'pump_turn_1st_derimax_min_sub': 1, 'engine_turnmax_min_sub': 1, 'pump_turnstd_mean_sub': 1, 'data_len': 4, '分配压力_max': 3, '油泵转速_1st_deri_fftstd_mean_sub': 1, '液压油温_fft_mean': 1, '流量档位_1st_deri_unique_len': 1, '排量电流_fft_max': 2, '液压油温_sum': 3, '发动机转速_sum': 1, '泵送压力_1st_derimax_min_sub': 1, '排量电流_max': 1, '油泵转速_mean': 1, '液压油温_1st_deri_fft_mean': 1, '排量电流_min': 1, 'pump_turn_sum': 1, 'pump_turn_1st_deri_max': 1, '泵送压力_1st_deri_max': 1, '液压油温_std': 1, '发动机转速_min': 1, '液压油温_min': 1, '分配压力_sum': 1}
```

* Local Score: 0.6021311711089291
  * [0.6040740114932209, 0.5983233874000753, 0.6058586415449023, 0.6103212835080378, 0.5920785315984092]
* Online Score: 0.59898978000

### version 5 (no improvement)

Feature added

* median

Result:

```txt
Feature Importance: ( 39 )
 {'发动机转速_1st_deri_fft_sum': 1, '泵送压力_fft_std': 4, '液压油温_fft_median': 1, '油泵转速_min': 1, 'engine_turn_min': 1, '分配压力std_mean_sub': 1, '液压油温_1st_deri_fft_sum': 4, '分配压力_median': 3, '排量电流_std': 1, '分配压力_min': 2, '分配压力_fft_mean': 3, '排量电流_1st_deri_sum': 1, '活塞工作时长_sum': 2, '油泵转速_fftmax_min_sub': 1, '分配压力_fft_sum': 1, '流量档位_fft_sum': 1, '泵送压力_1st_deri_fft_sum': 1, '液压油温_min': 1, '泵送压力std_mean_sub': 1, '高压开关_sum': 3, '发动机转速_1st_deri_fft_std': 1, '流量档位_unique_len': 1, '油泵转速_1st_deri_sum': 2, '排量电流_1st_deri_min': 1, 'data_drop_dup_len': 2, '低压开关_sum': 1, '排量电流_max': 1, '泵送压力_1st_deri_fft_min': 1, '分配压力_mean': 1, '发动机转速_max': 1, '分配压力_unique_len': 1, '泵送压力_median': 1, 'engine_turn_fftmax_min_sub': 1, '流量档位_median': 1, '发动机转速_1st_derimax_min_sub': 1, '排量电流_median': 1, '排量电流_sum': 1, '流量档位_1st_derimax_min_sub': 1, 'pump_turn_sum': 1}
```

* Local Score: 0.6035370675673373
  * [0.6034908706127333, 0.5964194713163171, 0.6076629016272395, 0.6100767627423971, 0.6000353315379994]
* Online Score: 0.59560806000

Change XGBoost parameter

* max_depth=8 (default=6)

```txt
Feature Importance: ( 98 )
 {'发动机转速_1st_deri_fft_sum': 1, '泵送压力_fft_std': 5, '液压油温_fft_median': 3, '油泵转速_min': 3, 'engine_turn_min': 1, '分配压力std_mean_sub': 1, 'pump_turn_1st_deri_median': 1, '油泵转速_std': 1, '液压油温_1st_deri_fft_sum': 5, '流量档位_mean': 2, 'pump_turn_1st_deri_fft_min': 1, '油泵转速std_mean_sub': 2, '分配压力_mean': 2, '分配压力_median': 8, '排量电流_std': 1, '分配压力_min': 2, '流量档位_min': 1, '发动机转速_min': 3, '分配压力_fft_mean': 4, '排量电流_1st_deri_sum': 1, '活塞工作时长_sum': 5, '油泵转速_fftmax_min_sub': 2, '分配压力_fft_sum': 1, '分配压力max_min_sub': 1, '分配压力_1st_deristd_mean_sub': 1, '油泵转速_1st_deri_fftmax_min_sub': 1, '泵送压力_fft_sum': 1, '油泵转速_1st_deri_std': 1, '流量档位_fft_sum': 2, 'data_drop_dup_len': 5, '发动机转速_std': 1, '泵送压力_1st_deri_fft_sum': 1, '液压油温_min': 2, 'engine_turn_fft_mean': 2, '油泵转速_fft_mean': 1, '高压开关_sum': 4, '油泵转速max_min_sub': 1, '泵送压力std_mean_sub': 1, 'pump_turn_fft_sum': 1, '分配压力_sum': 2, '发动机转速_median': 1, '液压油温_1st_deri_sum': 1, '泵送压力_1st_deristd_mean_sub': 2, '发动机转速_1st_deri_fft_std': 1, '泵送压力_mean': 1, '泵送压力_min': 1, '泵送压力_1st_deri_fftstd_mean_sub': 1, '液压油温_fft_max': 2, '发动机转速_1st_deri_sum': 2, '流量档位_unique_len': 1, '排量电流_median': 2, '分配压力_max': 7, '油泵转速_1st_deri_sum': 3, '排量电流_1st_deri_min': 1, 'pump_turn_1st_deri_fftstd_mean_sub': 1, '低压开关_sum': 1, '排量电流_max': 3, '泵送压力_1st_deri_fft_min': 2, '泵送压力_1st_derimax_min_sub': 1, '油泵转速_fft_std': 1, '液压油温_mean': 1, '分配压力_1st_deri_sum': 1, '发动机转速_max': 1, '分配压力_unique_len': 1, '液压油温_max': 1, '油泵转速_fftstd_mean_sub': 1, '分配压力_fftmax_min_sub': 1, '泵送压力_1st_deri_std': 1, '泵送压力_median': 1, 'pump_turn_1st_deri_fft_sum': 2, '发动机转速_fft_mean': 1, '流量档位_std': 1, '油泵转速_mean': 1, '油泵转速_1st_deri_mean': 1, 'engine_turnmax_min_sub': 1, 'engine_turn_fftmax_min_sub': 1, '流量档位_median': 1, '发动机转速_1st_deri_fftstd_mean_sub': 1, '排量电流_1st_deristd_mean_sub': 1, '排量电流_1st_deri_median': 1, '发动机转速_1st_derimax_min_sub': 2, '排量电流_1st_deri_unique_len': 1, '发动机转速_1st_deri_fftmax_min_sub': 1, '流量档位_max': 1, '油泵转速_1st_deri_fft_min': 1, '泵送压力_fftstd_mean_sub': 1, '泵送压力_fft_mean': 1, '排量电流_sum': 1, '流量档位_1st_derimax_min_sub': 1, '液压油温_1st_deri_fft_max': 1, '流量档位_1st_deri_sum': 1, '分配压力_std': 1, 'pump_turn_sum': 1, '分配压力_fft_median': 1, '发动机转速_1st_deri_std': 1, 'pump_turn_mean': 1, '流量档位_fft_min': 1, '液压油温std_mean_sub': 1}
```

* Local Score: 0.6072784415366662
  * [0.6070735360682296, 0.5997637232730834, 0.6125269064640029, 0.6099455986020119, 0.6070824432760031]
* Online Score: 0.59962279000

---

* max_depth=10 (default=6)

```txt
Feature Importance: ( 173 )
 {'发动机转速_1st_deri_fft_sum': 1, '泵送压力_fft_std': 6, '液压油温_fft_median': 6, '油泵转速_min': 4, 'engine_turn_min': 2, '分配压力std_mean_sub': 3, 'pump_turn_1st_deri_median': 2, '油泵转速_std': 2, '液压油温_1st_deri_fft_sum': 7, '流量档位_mean': 5, 'pump_turn_1st_deri_fft_min': 1, '油泵转速std_mean_sub': 4, '分配压力_mean': 6, '油泵转速_1st_derimax_min_sub': 2, '发动机转速_max': 8, '分配压力_min': 3, '分配压力_median': 14, '排量电流_std': 3, '流量档位_min': 1, '泵送压力_1st_deri_min': 2, '发动机转速_min': 7, '分配压力_fft_mean': 7, '排量电流_1st_deri_sum': 1, '活塞工作时长_sum': 8, '油泵转速_fftmax_min_sub': 2, '分配压力_fft_sum': 1, '分配压力max_min_sub': 1, '分配压力_1st_deristd_mean_sub': 3, '油泵转速_1st_deri_fftmax_min_sub': 1, '排量电流_1st_deristd_mean_sub': 3, '泵送压力_fft_sum': 1, '油泵转速_1st_deri_std': 1, '流量档位_fft_sum': 3, 'data_drop_dup_len': 9, '发动机转速_std': 2, '泵送压力_1st_deri_fft_sum': 1, '液压油温_min': 3, 'engine_turn_fft_mean': 2, '油泵转速_1st_deri_fft_min': 4, '发动机转速max_min_sub': 2, '油泵转速_fft_mean': 1, '泵送压力_1st_deristd_mean_sub': 5, '排量电流std_mean_sub': 5, '分配压力_1st_deri_fft_median': 1, '液压油温_std': 2, '高压开关_sum': 5, '排量电流_1st_deri_fftstd_mean_sub': 2, '分配压力_max': 13, '油泵转速max_min_sub': 3, '分配压力_fftstd_mean_sub': 1, '流量档位_fft_std': 2, '油泵转速_fftstd_mean_sub': 2, '流量档位_1st_deri_median': 1, '发动机转速_1st_deri_min': 1, '泵送压力std_mean_sub': 2, 'pump_turn_fft_sum': 1, 'engine_turn_std': 1, '泵送压力_min': 4, '流量档位_1st_deri_fftstd_mean_sub': 2, '分配压力_sum': 2, '流量档位_median': 4, '排量电流_1st_deri_min': 3, '流量档位_1st_deri_fft_min': 1, '发动机转速_median': 1, '液压油温_1st_deri_sum': 1, '排量电流_fft_sum': 1, '发动机转速_mean': 3, '发动机转速_1st_deri_fft_mean': 1, 'pump_turn_max': 1, '泵送压力_1st_deri_median': 3, '流量档位_fftstd_mean_sub': 2, '液压油温_1st_deri_fft_std': 2, '泵送压力max_min_sub': 1, 'pump_turn_std': 1, '发动机转速_1st_deri_fft_std': 2, '泵送压力_mean': 1, 'engine_turn_max': 1, '发动机转速_1st_deri_std': 2, '排量电流_max': 6, '流量档位_max': 3, '泵送压力_1st_deri_fftstd_mean_sub': 2, '流量档位_1st_deri_sum': 3, '排量电流_mean': 1, '液压油温_fft_max': 2, '发动机转速_1st_deri_sum': 2, '排量电流_fft_std': 3, 'engine_turn_fft_std': 1, '液压油温_1st_deri_fft_max': 2, '流量档位_unique_len': 1, '排量电流_median': 2, '液压油温_median': 1, '油泵转速_1st_deri_sum': 3, 'pump_turn_1st_deri_fftstd_mean_sub': 1, '发动机转速_1st_deri_median': 1, '流量档位_sum': 3, '低压开关_sum': 1, '泵送压力_1st_deri_fft_min': 2, '泵送压力_1st_derimax_min_sub': 1, '分配压力_1st_deri_unique_len': 1, '液压油温_max': 4, '油泵转速_fft_std': 1, '排量电流_1st_deri_unique_len': 2, '油泵转速_1st_deri_fft_max': 2, '发动机转速_1st_deristd_mean_sub': 1, '液压油温_mean': 2, '分配压力_1st_deri_sum': 2, '分配压力_unique_len': 1, '流量档位_1st_derimax_min_sub': 3, '发动机转速_sum': 3, '泵送压力_unique_len': 2, '液压油温_fft_sum': 2, '油泵转速_1st_deri_min': 2, '液压油温_fft_std': 2, '分配压力_fftmax_min_sub': 1, 'pump_turn_min': 2, 'pump_turn_mean': 2, '油泵转速_median': 2, '油泵转速_fft_min': 1, 'engine_turn_1st_deri_median': 1, '液压油温std_mean_sub': 2, '排量电流_sum': 2, 'engine_turn_1st_deri_fft_median': 1, '油泵转速_sum': 1, '排量电流_1st_deri_fft_sum': 1, '泵送压力_1st_deri_std': 1, '流量档位_1st_deri_mean': 1, '泵送压力_max': 2, '泵送压力_median': 3, 'pump_turn_1st_deri_fft_sum': 2, '发动机转速_fft_mean': 1, '发动机转速_1st_deri_fftstd_mean_sub': 2, '流量档位_std': 2, '分配压力_1st_deri_min': 1, '油泵转速_mean': 1, '油泵转速_1st_deri_mean': 2, '液压油温_1st_deri_median': 2, '液压油温_1st_deri_mean': 1, 'engine_turn_1st_deristd_mean_sub': 1, 'engine_turnmax_min_sub': 1, 'engine_turn_1st_deri_fftmax_min_sub': 1, '液压油温_1st_deri_fft_min': 3, '排量电流_1st_deri_median': 2, 'engine_turn_fftmax_min_sub': 1, '反泵_sum': 1, '流量档位_1st_deristd_mean_sub': 1, 'pump_turn_1st_deristd_mean_sub': 1, '排量电流_1st_deri_fft_min': 2, '发动机转速_fft_min': 1, '分配压力_1st_deri_fftmax_min_sub': 1, '油泵转速_fft_sum': 2, '发动机转速_1st_derimax_min_sub': 2, '液压油温_1st_deri_fftstd_mean_sub': 1, 'engine_turn_1st_deri_max': 1, '发动机转速_1st_deri_fftmax_min_sub': 1, '发动机转速_unique_len': 3, '排量电流_min': 3, 'pump_turn_1st_deri_fft_median': 1, 'pump_turn_median': 1, '泵送压力_1st_deri_sum': 2, '泵送压力_fftstd_mean_sub': 1, '液压油温_1st_deri_max': 2, '发动机转速_1st_deri_fft_max': 2, '泵送压力_fft_mean': 1, '泵送压力_1st_deri_fft_std': 1, '油泵转速_unique_len': 1, '泵送压力_std': 1, 'pump_turn_sum': 2, '分配压力_1st_deri_fft_sum': 1, '分配压力_std': 1, '分配压力_fft_median': 1, '分配压力_1st_deri_fft_min': 1, '泵送压力_1st_deri_fft_max': 1, '流量档位_fft_min': 1}
```

* Local Score: 0.6054278544166276
  * [0.6052623916625964, 0.6049232168131269, 0.6105130573312874, 0.6006873440859036, 0.6057532621902235]
* Online Score: 0.59613919000

---

Change XGBoost parameter

* max_bin=2^(10-1) (default=256)

> not sure why it isn't `max_leaves`...

No more improvement...

### version 6 (TODO)

* [tsfresh](https://tsfresh.readthedocs.io/en/v0.1.2/index.html)
  * [blue-yonder/tsfresh](https://github.com/blue-yonder/tsfresh)

## TODO

* [ ] Matplotlib Chinese font /or/ Transfer to English label
  * [matplotlib - Configuring the font family](https://matplotlib.org/gallery/api/font_family_rc_sgskip.html)
  * [How to Plot Unicode Characters with Matplotlib](https://jdhao.github.io/2018/04/08/matplotlib-unicode-character/)
  * [CSDN - Python Matplot中文顯示完美解決方案](https://blog.csdn.net/kesalin/article/details/71214038)
    * matplotlib/font_manager.py
        > Matplotlib is building the font cache using fc-list
* [ ] Add new feature based on previous created feature pickle. (skip the feature with same name)
* [ ] Replace the scikit learn f1_score with customized @jit f1 score function

## Links

* [**週冠軍分享|2019 DCIC 《混凝土泵車砼活塞故障預警》**](https://mp.weixin.qq.com/s?__biz=MzI5ODQxMTk5MQ==&mid=2247485857&idx=2&sn=148ad7cb473f75bf6719c46f1f75dfd6&chksm=eca77b19dbd0f20fb21cb6959383b6bbe942f256f5a03e5ed870ad463beeb91e247643d7874a&mpshare=1&scene=23&srcid=#rd)
* [**天線寶寶's baseline - jmxhhyx/DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles**](https://github.com/jmxhhyx/DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles)
* [**tianshuaifei/dcic_2019**](https://github.com/tianshuaifei/dcic_2019)
* [abanger/DCIC2019-Concrete-Pump-Vehicles](https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles)

### Packages

* [sklearn - Model persistence (joblib)](https://scikit-learn.org/stable/modules/model_persistence.html)
* [tqdm](https://github.com/tqdm/tqdm) - A fast, extensible progress bar for Python and CLI
