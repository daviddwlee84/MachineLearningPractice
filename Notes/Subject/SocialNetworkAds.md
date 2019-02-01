# Social Network Ads

## Dataset

[Social Network Ads](https://www.kaggle.com/rakeshrau/social-network-ads)

## Result

### LightGBM

#### First Attempt

Parameter

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

```txt
[LightGBM] [Info] Number of positive: 111, number of negative: 189
[LightGBM] [Info] Total Bins 111
[LightGBM] [Info] Number of data: 300, number of used features: 2
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.370000 -> initscore=-0.532217
[LightGBM] [Info] Start training from score -0.532217
```

```txt
Confusion Matrix:
 [[68  0]
 [32  0]]
Accuracy: 0.68
```

### Second attempt

```txt
[LightGBM] [Warning] Starting from the 2.1.2 version, default value for the "boost_from_average" parameter in "binary" objective is true.
This may cause significantly different results comparing to the previous versions of LightGBM.
Try to set boost_from_average=false, if your old models produce bad results
```

Parameter

```py
params = {
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 10,
    'min_data': 50,
    'max_depth': 10,
    'boost_from_average': False
}
```

```txt
[LightGBM] [Info] Number of positive: 111, number of negative: 189
[LightGBM] [Info] Total Bins 111
[LightGBM] [Info] Number of data: 300, number of used features: 2
```

```txt
Confusion Matrix:
 [[55 13]
 [ 2 30]]
Accuracy: 0.85
```
