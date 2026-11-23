# Feature Engineering

[Data Preprocessing](MachineLearningConcepts.md#Data-Preprocessing)

## Concept of Data

* **Structured data**
  * numeric data
  * catagorical data
* **Non-structured data**
  * text
  * image
  * sound
  * video

## Tips for Creating Baseline

> [Baseline Model | Kaggle](https://www.kaggle.com/matleonard/baseline-model)

## Tips for Structured Data

### Tips for Categorical Feature

> [Categorical Encodings | Kaggle](https://www.kaggle.com/matleonard/categorical-encodings#Count-Encoding)

* [Sample categorical feature encoding methods](https://www.kaggle.com/c/avito-demand-prediction/discussion/55521)
  * label-encode
  * mean-encoding (be careful, easy over fitting if you have not a right validation strategy)
  * factorize-encoding
  * frequency-encoding

* [Cat2Vec (paper pdf)](https://openreview.net/pdf?id=HyNxRZ9xg) - LEARNING DISTRIBUTED REPRESENTATION OF MULTI-FIELD CATEGORICAL DATA

Tools

* [scikit-learn-contrib/categorical-encoding: A library of sklearn compatible categorical variable encoders](https://github.com/scikit-learn-contrib/categorical-encoding)

#### (Category) Encoding

* Ordinal Encoding (序號編碼)
  * `encoder = sklearn.preprocessing.LabelEncoder()`
    * `df[cols].apply(encoder.fit_transform)`
    * `encoder.fit_transform(df[col])`
* One-Hot (benefit for linear models or NN)
  * use sparse vector to reduce space
  * co-operate with feature selection to reduce dimension
    * if dimension is too high => hard to calculate distance, and easy to overfit due to too many parameters
  * Python
    * [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)
* Binary Encoding

#### Count Encoding

* Count encoding replaces each categorical value with the number of times it appears in the dataset.

> * Why is count encoding effective?
>
>   Rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping.

#### Target Encoding

* Target encoding replaces a categorical value with the average value of the target for that value of the feature.
* This technique uses the targets to create new features. So including the validation or test data in the target encodings would be a form of target leakage. Instead, you should learn the target encodings from the training dataset only and apply it to the other datasets.

> Sometimes some feature don't profit from Target Encoding (rather, it reduce the accuracy)
>
> Target encoding attempts to measure the population mean of the target for each level in a categorical feature. This means when there is less data per level, the estimated mean will be further away from the "true" mean, there will be more variance. There is little data per IP address so it's likely that the estimates are much noisier than for the other features. The model will rely heavily on this feature since it is extremely predictive. This causes it to make fewer splits on other features, and those features are fit on just the errors left over accounting for IP address. So, the model will perform very poorly when seeing new IP addresses that weren't in the training data (which is likely most new data). Going forward, we'll leave out the IP feature when trying different encodings.

#### CatBoost Encoding

* This is similar to target encoding in that it's based on the target probablity for a given value. However with CatBoost, for each row, the target probability is calculated only from the rows before it.

#### Leave-one-out Encoding

#### Feature embedding with SVD

### Tips for Numeric/Continuous Feature

* mean
* sum
* max
* min
* std
* var
* np.ptp
* ... TBD

#### Transforming numerical features - Deal with long tail data

* sqrt
* log
  * The log transformation won't help tree-based models since tree-based models are scale invariant. However, this should help if we had a linear model or neural network.

## Tips for Non-structured Data

### Tips for Text Data

[Information Retrieval - Text Representation](Information_Retrieval.md#Text-Representation)

* Bag-of-Words + TF-IDF & N-gram

* Topic Model
  * A model based on probability graph => generative model
  * Infer the latent variable (i.e. the topic)
  * Example
    * LDA
* Word Embedding
  * Using neural network model
  * Example
    * Word2Vec

#### Word Stemming

Changing a different type words into same word

e.g. speek, spoke, spoken => speek

#### Word2Vec vs. LDA

* Word2Vec: Use neural network to generate a dense representation of a word
  * can be considered as learning the "context-word" matrix
  * model
    * CBOW (Continues Bag of Words)
    * Skip-gram
* LDA: Use the "showing-at-the-same-time" relationship to cluster word with "topic"
  * can be considered as factorize the "document-word" matrix

### Tips for Image Data

Prior hypothesis/information

* on model
  * Transfer Learning
* on condition
* on constrain
* on dataset
  * Data Augmentation

Model

* Transfer Learning + Fine-tune
* GAN

#### Data Augmentation

CV

1. rotate, shift, scale, cut, fill, flip, ...
2. add noise
3. change color
4. change brightness, clarity, contrast, sharpness

* [albu/albumentations: fast image augmentation library and easy to use wrapper around other libraries](https://github.com/albu/albumentations)

## Tips for Special Types of Data

### Tips for Count Feature

* Unique length
* Data length
* ...

### Tips for Time Series Feature

#### First Derivative

#### Descrete Fourier Transform

* [What FFT descriptors should be used as feature to implement classification or clustering algorithm?](https://stackoverflow.com/questions/27546476/what-fft-descriptors-should-be-used-as-feature-to-implement-classification-or-cl)
* [numpy.fft](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html)

tsfresh

* [tsfresh document](https://tsfresh.readthedocs.io/en/latest/index.html)
* [tsfresh github](https://github.com/blue-yonder/tsfresh)

## Tips for Getting New Feature

> [Feature Generation | Kaggle](https://www.kaggle.com/matleonard/feature-generation)

Model may not extract some message like human does. We can build new feature by combining other features to gain accuracy.

* Think about business situation
* Learn from the open source kernel and the discussion session

### Interaction, Feature Combination

> combining categorical variables to create more categorial features

* Naive combine
* Combine and represent with lower dimension

This is a new categorical feature that can provide information about correlations between categorical variables. This type of feature is typically called an interaction. In general, you would build interaction features from all pairs of categorical features. You can make interactions from three or more features as well, but you'll tend to get diminishing returns.

> `itertools.combinations`
>
> ```py
> # Iterate through all the pair of features
> for col1, col2 in itertools.combinations(columns, 2)
> ```

## Tips for Feature Selection

> [Feature Selection | Kaggle](https://www.kaggle.com/matleonard/feature-selection)

### What to define a good feature

* This feature gain a lot of improvement
* **Feature importance**
  * LightGBM
    * lgb.plot_importance(lgb_model, max_num_features=10)
    * feature_name = lgb_model.feature_name()
    * importance = lgb_model.feature_importance(importance_type='split')
  * XGBoost
    * xgb.plot_importance(xgb_model, max_num_features=25)
    * xgb_model.get_fscore()

### Univariate Feature Selection

> Univariate methods consider only one feature at a time when making a selection decision

The simplest and fastest methods are based on univariate statistical tests.

For each feature, measure how strongly the target depends on the feature using a statistical test like $\chi^2$ or ANOVA.

* ANOVA F-value
  * The F-value measures the linear dependency between the feature variable and the target.
  * This means the score might underestimate the relation between a feature and the target if the relationship is nonlinear.
* Mutual Information Score
  * The mutual information score is nonparametric and so can capture nonlinear relationships.

> The best value of $k$:
>
> To find the best value of K, you can fit multiple models with increasing values of K, then choose the smallest K with validation score above some threshold or some other criteria. A good way to do this is loop over values of K and record the validation scores for each iteration.

* sklearn
  * `sklearn.feature_selection.SelectKBest`: returns the K best features given some scoring function
    * For our classification problem, the module provides three different scoring functions: $\chi^2$ , ANOVA F-value, and the mutual information score.
      * define the number of features to keep, based on the score from the scoring function
  * `sklearn.feature_selection.f_classif`

### L1 Regularization

Make our selection using all of the features by including them in a linear model with L1 regularization. This type of regularization (sometimes called Lasso) penalizes the absolute magnitude of the coefficients, as compared to L2 (Ridge) regression which penalizes the square of the coefficients.

As the strength of regularization is increased, features which are less important for predicting the target are set to 0. This allows us to perform feature selection by adjusting the regularization parameter. We choose the parameter by finding the best performance on a hold-out set, or decide ahead of time how many features to keep.

> Top K features with L1 regularization
>
> Here you've set the regularization parameter `C=0.1` which led to some number of features being dropped. However, by setting `C` you aren't able to choose a certain number of features to keep.
>
> To select a certain number of features with L1 regularization, you need to find the regularization parameter that leaves the desired number of features. To do this you can iterate over models with different regularization parameters from low to high and choose the one that leaves K features.
> Note that for the scikit-learn models C is the inverse of the regularization strength.

* sklearn
  * `sklearn.feature_selection.SelectFromModel`: to select the non-zero coefficients
    * used along with
      * `sklearn.linear_model.Lasso`
      * `sklearn.linear_model.LogisticRegression`

### Feature Selection with Trees

use something like `RandomForestClassifier` or `ExtraTreesClassifier` to find feature importances

### Summary of Feature Selection

In general, feature selection with L1 regularization is more powerful the univariate tests, but it can also be very slow when you have a lot of data and a lot of features. Univariate tests will be much faster on large datasets, but also will likely perform worse.

## Tips for Memory Usage

Feature engineering will need lots of RAM (usually > 16GB).

### Pandas

#### astype

Most of the time, you won't need very precise data. Use not-so-precise data type will save memory and time.

* float: float32 (default float64)
* string: int32

#### [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

#### [dataframe.groupby.agg](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html)

* [Summarising, Aggregating, and Grouping data in Python Pandas](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/)

- np.ptp (peak to peak)
  - Range of values (maximum - minimum) along an axis.

#### [pandas.Series.dt.month](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.month.html)

### Pickle

#### MacOS file large than 4GB

* [stackoverflow](https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb)

```py
def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
```

## Tips for Fast Evaluation

### [Compiling Python code with @jit](http://numba.pydata.org/numba-doc/0.17.0/user/jit.html)

## Tips for parameter

### Gain parameter

#### Bayesian Optimization

#### Optuna

#### Hyperopt

### Test parameter

#### K-Fold Cross-Validation

## Tips for Train/Valid/Test Split

### For Time Series Data

* Since our model is meant to predict events in the future, we must also validate the model on events in the future. If the data is mixed up between the training and test sets, then future data will leak in to the model and our validation results will overestimate the performance on new data.

```py
valid_fraction = 0.1
valid_size = int(len(data) * valid_fraction)

train = data[:-2 * valid_size] # 80%
valid = data[-2 * valid_size:-valid_size] # 10%
test = data[-valid_size:] # 10%
```

### Data Leakage

> [Data Leakage | Kaggle](https://www.kaggle.com/alexisbcook/data-leakage)

* You should calculate the encodings from the training set only. If you include data from the validation and test sets into the encodings, you'll overestimate the model's performance. You should in general be vigilant to avoid leakage, that is, including any information from the validation and test sets into the model.

## Tips for Models

### Model Selection

#### Tree-based vs Neural Network Models

The features themselves will work for either model. However, numerical inputs to neural networks need to be standardized first. That is, the features need to be scaled such that they have 0 mean and a standard deviation of 1. This can be done using `sklearn.preprocessing.StandardScaler`.

### Candidate Model Options

#### Classifier

* LightGBM
  * LightGBM models work with label encoded features, so you don't actually need to one-hot encode the categorical features.
* Nerual Network

#### Regression

## Resources

* [**分分鐘帶你殺入Kaggle Top 1% - 知乎**](https://zhuanlan.zhihu.com/p/27424282)
* [如何ensemble多個神經網路？ - 知乎](https://www.zhihu.com/question/60753512/answer/184409655)
* [**Kaggle Ensembling Guide | MLWave**](https://mlwave.com/kaggle-ensembling-guide/)
* [Wiki - Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)
* [機器學習筆記 - 特徵工程](https://feisky.xyz/machine-learning/basic/feature-engineering.html)
* [**Understanding Feature Engineering (Part 1) — Continuous Numeric Data**](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)

### Tutorial

* [**Feature Engineering | Kaggle**](https://www.kaggle.com/learn/feature-engineering)

### Good Example

* [Data Fountain光伏發電量預測 Top1 開源分享](https://zhuanlan.zhihu.com/p/44755488?utm_source=qq&utm_medium=social&utm_oi=623925402599559168)
