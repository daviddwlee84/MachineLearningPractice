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
Confusion Matrix:
 [[68  0]
 [32  0]]
Accuracy: 0.68
```
