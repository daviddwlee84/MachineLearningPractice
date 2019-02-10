# Kaggle Elo Merchant Category Recommendation

## Competition

* [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation)

### Description

Imagine being hungry in an unfamiliar part of town and getting restaurant recommendations served up, based on your personal preferences, at just the right moment. The recommendation comes with an attached discount from your credit card provider for a local place around the corner!

Right now, [Elo](https://www.cartaoelo.com.br/), one of the largest payment brands in Brazil, has built partnerships with merchants in order to offer promotions or discounts to cardholders. But do these promotions work for either the consumer or the merchant? Do customers enjoy their experience? Do merchants see repeat business? Personalization is key.

Elo has built machine learning models to understand the most important aspects and preferences in their customers’ lifecycle, from food to shopping. But so far none of them is specifically tailored for an individual or profile. This is where you come in.

In this competition, Kagglers will develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. Your input will improve customers’ lives and help Elo reduce unwanted campaigns, to create the right experience for customers.

### Data

`train.csv`

Columns|Description
-------|-----------
card_id|Unique card identifier
first_active_month|'YYYY-MM', month of first purchase
feature_1|Anonymized card categorical feature
feature_2|Anonymized card categorical feature
feature_3|Anonymized card categorical feature
target|Loyalty numerical score calculated 2 months after historical and evaluation period

`historical_transactions.csv` and `new_merchant_period.csv`

> these two files contain the same variable
> and the difference between the two tables
> only concern the position with respect to a reference date

Columns|Description
-------|-----------
card_id|Card identifier
month_lag|month lag to reference date
purchase_date|Purchase date
authorized_flag|Y' if approved, 'N' if denied
category_3|anonymized category
installments|number of installments of purchase
category_1|anonymized category
merchant_category_id|Merchant category identifier (anonymized)
subsector_id|Merchant category group identifier (anonymized)
merchant_id|Merchant identifier (anonymized)
purchase_amount|Normalized purchase amount
city_id|City identifier (anonymized)
state_id|State identifier (anonymized)
category_2|anonymized category

`merchants.csv`

Columns|Description
-------|-----------
merchant_id|Unique merchant identifier
merchant_group_id|Merchant group (anonymized )
merchant_category_id|Unique identifier for merchant category (anonymized )
subsector_id|Merchant category group (anonymized )
numerical_1|anonymized measure
numerical_2|anonymized measure
category_1|anonymized category
most_recent_sales_range|Range of revenue (monetary units) in last active month --> A > B > C > D > E
most_recent_purchases_range|Range of quantity of transactions in last active month --> A > B > C > D > E
avg_sales_lag3|Monthly average of revenue in last 3 months divided by revenue in last active month
avg_purchases_lag3|Monthly average of transactions in last 3 months divided by transactions in last active month
active_months_lag3|Quantity of active months within last 3 months
avg_sales_lag6|Monthly average of revenue in last 6 months divided by revenue in last active month
avg_purchases_lag6|Monthly average of transactions in last 6 months divided by transactions in last active month
active_months_lag6|Quantity of active months within last 6 months
avg_sales_lag12|Monthly average of revenue in last 12 months divided by revenue in last active month
avg_purchases_lag12|Monthly average of transactions in last 12 months divided by transactions in last active month
active_months_lag12|Quantity of active months within last 12 months
category_4|anonymized category
city_id|City identifier (anonymized )
state_id|State identifier (anonymized )
category_2|anonymized category

### Evaluation

Root Mean Squared Error (RMSE)

## Directory Structure

* `raw_data`: data download from kaggle
* `data`: preprocessed pickle data
* `deal`: filter/project part of the data
* `feat`: features we've made
* `model`: LightGBM model
* `prediction`: prediction result

- `feature.py`: building features
- `train_model.py`: training model
  - online
    - 0: 5-fold cross-validation
    - 1: train model + predict result

## Usage

### Downloading Dataset

Put csv files into `raw_data/`

* historical_transactions.csv  
* merchants.csv
* new_merchant_transactions.csv
* train.csv
* test.csv

### Feature Engineering

```sh
python3 feature.py
```

### Training Model

```sh
python3 train_model.py
```

## Others Kaggle Kernel Collection

### Knowing data distribution

https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
https://www.kaggle.com/lokendradevangan/elo-recommendation-eda

### Good feature engineering example

* [Elo world](https://www.kaggle.com/fabiendaniel/elo-world) - Referenced
* [LGB + FE (LB 3.707)](https://www.kaggle.com/konradb/lgb-fe-lb-3-707)
