# LightGBM

## Overview

### Differs from other tree-based algorithm

* LightGBM
  * grows tree vertically -> grows tree leaf-wise
  * it will choose the leaf with max delta loss to grow
  * when growing the same leaf, leaf-wise algorithm can reduce more loss than a leve-wise algorithm
* Other tree-based algorithm
  * grows trees horizontally -> grows level-wise

### Pros and Cons of LightGBM

Pros

* high speed (prefix "Light")
* handle the large size of data and takes lower memory to run
* focuse on accuracy of results
* support GPU learning

Cons

* sensitive to overfitting (can easily overfit small data)

> better use it only for data with 10000+ rows

### Implementation of LightGBM

Implementation of LightGBM is easy, the only complicated thing is **parameter tuning**. (LightGBM covers more than 100 parameters)

## Parameter Tuning

Used to improve the model efficiency

* num_leaves
  * main parameter to control the complexity of the tree model
  * should be <= $2^{\text{max depth}}$, value more than this will result in overfitting
* min_data_in_leaf
  * Setting it to a large value can avoid growing too deep, but may cause underfitting
  * In practice, setting it to hundreds or thousands is enough for a large dataset
* max_depth

For faster speed

* Use bagging
  * setting `bagging_fraction`, `bagging_freq`
* Use feature sub-sampling
  * setting `feature_fraction`
* Use small `max_bin`
* Use `save_binary` to seed up data loading in future learning
* Use parallel learning

For better accuracy

* Use large `max_bin`
* Use small `learning_rate` with large `num_iterations`
* Use large `num_leaves` (may cause overfitting)
* Use bigger training data
* Try `dart`
* Try to use categorical feature directly

To deal with overfitting

* Use small `max_bin`
* Use small `num_leaves
* Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
* Use bagging
  * setting `bagging_fraction`, `bagging_freq`
* Use feature sub-sampling
  * setting `feature_fraction`
* Use bigger training data
* Try regularization with `lambda_l1`, `lambda_l2` and `min_gain_to_split`
* Try `max_depth` to avoid growing deep tree

### Parameters

#### Control Parameters

Parameter|Description|Main Usage
---------|-----------|----------
max_depth|maximum depth of tree|handle model overfitting (-> lower it)
min_data_in_leaf|minimum number of the records a leaf may have (default 20)|deal overfitting
feature_fraction|select a fraction of feature randomly in each iteration for building tree (default 0.8)|be used in "random forest boosting"
bagging_fraction|specifies the fraction of data to be used for each iteration|speed up the training and avoid overfitting
early_stopping_round|model will stop training if one metric of one validation data doesn't improve in last early_stopping_round rounds|speed up analysis, reduce excessive iterations
lambda|value range from 0~1|specifies regularization
min_gain_to_split|minimum gain to make a split|control number of useful splits in tree
max_cat_group|merge categories into 'max_cat_group' groups, finds the split points on the group boundaries|when the number of category is large -> merge them to prevent overfitting

#### Core Parameters

Parameter|Description|Option
---------|-----------|----------
Task|specifies the task you want to perform on data|train, predict
application|(most important) specifies the application of the model|regression (default), binary, multiclass
boosting|the type of algorithm|gbdt (default), rf, dart, goss
num_boost_round|number of boosting iteration|(typically 100+)
learning_rate|impact of each tree on the final outcome, controls the magnitude of this change in the estimate|typical values: 0.1, 0.001, 0.003...
num_leaves|number of leaves in full tree|(default 31)
device||cpu, gpu

#### Metric Parameters

specifies loss for model building (one of the important parameter)

* mae: mean absolute error
* mse: mean squared error
* binary_logloss: loss for binary classification
* multi_logloss: loss for multi classification

#### IO Parameters

Parameter|Description
---------|-----------
max_bin|maximum number of bin that feature value will bucket in
categorical_feature|the index of categorical features (i.e. the column number)
save_binary|"True" will save the dataset to binary file -> speed data reading time for next time

## Implementation

Data preprocessing

* Importing the dataset
* Splitting the dataset into the training set and test set
* Feature scaling

Model building

```py
import lightgbm as lgb
d_train = lgb.Dataset(data_train, label=label_train)
```

Set parameters in a dict

> * Use 'binary' as objective (because classification)
> * Use 'binary_logloss' as metric (because classification)
> * 'num_leaves' = 10 (as it's small data)
> * 'boosting type' is GBDT (it's okay to try Random Forest)

```py
params = {
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 10,
    'min_data': 50,
    'max_depth': 10
}
```

Training

```py
clf = lgb.train(params, d_train, 100)
```

Prediction

```py
y_pred = clf.predict(data_test)

# Convert into binary values
# value >= 0.5: 1
# value <  0.5: 0
```

Check result

```py
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
```

## Links

### Tutorial

* [Youtube - 數據科學家常用工具XGBoost與LightGBM大比拼，性能與結構](https://youtu.be/dOwKbwQ97tI)

### Article

* [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)

### Github

* [Microsoft/LightGBM](https://github.com/Microsoft/LightGBM)
