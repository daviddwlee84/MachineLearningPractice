import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
import feature

version = 'version'

deal_path = 'deal/'
feat_path = 'feat/'
model_path = 'model/'
prediction_path = 'prediction/'

def modeling(X, Y, categorical, online):
    seed = 333
    EARLY_STOP = 300
    OPT_ROUNDS = 691
    MAX_ROUNDS = 3000
        
    params = {
        'boosting': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'learning_rate': 0.01,
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin': 255,
        'subsample': 0.85,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'num_leaves':63,
        'seed': seed,
        'nthread': 8
    }
    
    if online == 0:
        print("Start train and validate...")
        
        dtrain = lgb.Dataset(X, label=Y, feature_name=list(X.columns), categorical_feature=categorical)
        
        eval_hist = lgb.cv(params, 
                           dtrain, 
                           nfold = 5,
                           num_boost_round=MAX_ROUNDS,
                           early_stopping_rounds=EARLY_STOP,
                           verbose_eval=50, 
                           seed = seed,
                           stratified = False
                          )
        
        print(eval_hist)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed, test_size=0.25)
        dtrain = lgb.Dataset(X_train,
                         label=Y_train,
                         feature_name=list(X.columns),
                         categorical_feature=categorical)
        
        dtest = lgb.Dataset(X_test, label=Y_test,
                            feature_name=list(X.columns),
                            categorical_feature=categorical)

        model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                      valid_sets=[dtrain,dtest], valid_names=['train', 'valid'],
                      early_stopping_rounds=EARLY_STOP, verbose_eval=20)
            
        print('OPT_ROUNDS:', model.best_iteration)
    else:
        print('Start training') # Be aware of setting OPT-ROUNDS
        dtrain = lgb.Dataset(X,
                             label=Y,
                             feature_name=list(X.columns),
                             categorical_feature=categorical)
        model = lgb.train(params,dtrain, num_boost_round=OPT_ROUNDS, valid_sets=[dtrain], valid_names=['train'], verbose_eval=100)

         
        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})
    
        importances.sort_values('importances', ascending=False, inplace=True)
    
    
        model.save_model(model_path + '{}.model'.format(version))
        importances.to_csv(model_path + '{}_importances.csv'.format(version), index=False)
    
        return model

def merge_feature():
    """ Merge training and test data with pre-built feature """
    # Loading original training and test data
    data_train = pd.read_pickle(deal_path + 'train.pickle')
    data_test = pd.read_pickle(deal_path + 'test.pickle')

    # Merge features to original data
    # load features
    feat_authorized_mean = pd.read_pickle(feat_path + 'authorized_mean.pickle')
    feat_historical_transactions = pd.read_pickle(feat_path + 'historical_transactions.pickle')
    feat_authorized_transactions = pd.read_pickle(feat_path + 'authorized_transactions.pickle')
    feat_new_merchant_transactions = pd.read_pickle(feat_path + 'new_merchant_transactions.pickle')
    feat_final_group = pd.read_pickle(feat_path + 'final_group.pickle')
    feat_additional_fields = pd.read_pickle(feat_path + 'additional_fields.pickle')
    
    # merge features
    data_train = pd.merge(data_train, feat_authorized_mean, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_authorized_mean, on='card_id', how='left')

    data_train = pd.merge(data_train, feat_historical_transactions, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_historical_transactions, on='card_id', how='left')

    data_train = pd.merge(data_train, feat_authorized_transactions, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_authorized_transactions, on='card_id', how='left')

    data_train = pd.merge(data_train, feat_new_merchant_transactions, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_new_merchant_transactions, on='card_id', how='left')

    data_train = pd.merge(data_train, feat_final_group, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_final_group, on='card_id', how='left')

    data_train = pd.merge(data_train, feat_additional_fields, on='card_id', how='left')
    data_test = pd.merge(data_train, feat_additional_fields, on='card_id', how='left')

    return data_train, data_test

def get_features():
    train_data, test_data = merge_feature()
    categorical = ['feature_1', 'feature_2', 'feature_3']
    return train_data, test_data, categorical
    
def train_and_predict(online):
    train_data, test_data, categorical = get_features()    
    train_label = train_data['target']
    train_data = train_data.drop(['card_id', 'target'], axis=1)
    pred_id = test_data['card_id']
    
    test_data = test_data.drop('card_id', axis=1)
    print('features:', train_data.columns)
    
    model = modeling(train_data, train_label, categorical, online)
        
    # model = lgb.Booster(model_file='model/09111902.model')
    # It's able to load the trained model
    # Write a document to record the status os training data
    
    if online == 1: # Submission
        preds = model.predict(test_data, num_iteration=model.best_iteration)
        result = pd.DataFrame({'card_id': pred_id, 'target': preds}, columns=['card_id', 'target'])
        result.to_csv(prediction_path + 'submit_{}.csv'.format(version), index=False)

if __name__=='__main__':
    version = datetime.now().strftime("%m%d%H%M")
    train_and_predict(1)
