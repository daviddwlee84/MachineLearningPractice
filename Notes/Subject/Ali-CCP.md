# Ali-CCP：Alibaba Click and Conversion Prediction

## Dataset

* [Ali-CCP：Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/datalab/dataSet.html?spm=5176.100073.0.0.2d186fc1w0MXC3&dataId=408)
* [Papper - Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://dl.acm.org/citation.cfm?id=3210104)

## Heuristic

Based on the given *Description of feature sets* table

1. Find similar (exact same (because of the size of dataset is too big)) user by User Features
2. Using this user info to build a prediction engine
3. In the test set, we assume we build a recommend mechanism that we can use a pop-up window to force user in test set to click it.
4. If in the test set the user actually click and buy that kind of goods. Then we say it's a good prediction (recommendation)
