import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from datetime import datetime

data_path = 'data/'
deal_path = 'deal/'
feat_path = 'feat/'
model_path = 'model/'

def init_data():
    data_train = pd.read_csv('raw_data/train.csv')
    data_test = pd.read_csv('raw_data/test.csv')
    data_history = pd.read_csv('raw_data/historical_transactions.csv')
    data_merchants = pd.read_csv('raw_data/merchants.csv')
    data_new_merchant_transactions = pd.read_csv('raw_data/new_merchant_transactions.csv')
    
    data_train.to_pickle(data_path + 'train.pickle')
    data_test.to_pickle(data_path + 'test.pickle')
    data_history.to_pickle(data_path + 'historical_transactions.pickle')
    data_merchants.to_pickle(data_path + 'merchants.pickle')
    data_new_merchant_transactions.to_pickle(data_path + 'new_merchant_transactions.pickle')

def deal_data():
    # In the "baseline", we only use projection part of categorical type date in new_merchant recently.
    deal_new_merchant_transactions = pd.read_pickle(data_path + 'new_merchant_transactions.pickle')
    cols = ['category_1','category_2','category_3']
    for col in cols:
        series = deal_new_merchant_transactions[col]
        cats = sorted(list(set(series.dropna().unique())))
        deal_new_merchant_transactions[col] = pd.Categorical(series, categories=cats).codes + 6
        deal_new_merchant_transactions[col] = deal_new_merchant_transactions[col].astype('int8')
    deal_new_merchant_transactions.to_pickle( deal_path + 'new_merchant_transactions.pickle')

def feature_new_merchants_category123(data):
    res = data['card_id'].drop_duplicates().sort_values().reset_index().drop('index', axis=1)
    
    cols = ['category_1', 'category_2', 'category_3']
    for col in cols:
        t = data.groupby('card_id')[col].agg(lambda x: stats.mode(x)[0][0]).reset_index()
        t = t.rename(columns = {col: 'mode_' + col})
        res = pd.concat([res, t['mode_' + col]], axis=1)
    print(res.info())
    res.to_pickle(feat_path + 'new_merchants_category123.pickle')

def feature_new_merchants_amountx(data):
    res = data['card_id'].drop_duplicates().sort_values().reset_index().drop('index', axis=1)
    
    cols = ['installments', 'purchase_amount']
    for col in cols:
        tags = ['sum', 'mean', 'max', 'min', 'std', 'median']
        t = data.groupby('card_id')[col].agg(tags).reset_index()
        t = t.rename( columns = {'sum':'sum_purchase_amount', 'mean':'mean_purchase_amount', 'max':'max_purchase_amount',\
                                 'min':'min_purchase_amount', 'std':'std_purchase_amount', 'median':'median_purchase_amount'})
        res = pd.concat([res,t[[x + '_purchase_amount' for x in tags]]],axis=1)
    print(res.info())
    res.to_pickle(feat_path+'new_merchants_amountx.pickle')

def merge_feature():
    model_train = pd.read_pickle(data_path + 'train.pickle')
    model_test = pd.read_pickle(data_path + 'test.pickle')
    
    feat_new_merchants_category123 = pd.read_pickle(feat_path + 'new_merchants_category123.pickle')
    model_train = pd.merge(model_train, feat_new_merchants_category123, how='left', on=['card_id'])
    model_test = pd.merge(model_test, feat_new_merchants_category123, how='left', on=['card_id'])
    
    feat_new_merchants_amountx = pd.read_pickle(feat_path + 'new_merchants_amountx.pickle')
    model_train = pd.merge(model_train,feat_new_merchants_amountx, how='left',on=['card_id'])
    model_test = pd.merge(model_test,feat_new_merchants_amountx, how='left',on=['card_id'])
    
    model_train = model_train.drop(['first_active_month'], axis=1)
    model_test = model_test.drop(['first_active_month'], axis=1)

    return model_train, model_test

if __name__=='__main__':
    version = datetime.now().strftime("%m%d%H%M")
    init_data()
    deal_data()
    data_new_merchant_transactions = pd.read_pickle('deal/new_merchant_transactions.pickle')
    feature_new_merchants_category123(data_new_merchant_transactions[['card_id', 'category_1', 'category_2', 'category_3']])
    feature_new_merchants_amountx(data_new_merchant_transactions[['card_id', 'installments', 'purchase_amount']])
#    merge_feature()
