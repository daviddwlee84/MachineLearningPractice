# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm # A fast, extensible progress bar for Python and CLI 
import pickle
import os
import xgboost as xgb

from sklearn.externals import joblib # For model persistance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

version = "v2"

data_dir = "raw_data/"
feature_dir = "feature/"
model_dir = "model/model_{}/".format(version)
submit_dir = "submit/"

## Utility Function

# Pickle
def save_variable_to_pickle(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)

def load_variable_from_pickle(filename):
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r

# F1 score validation (for XGBoost)
# 2019/3/7 : Use this will get lower f1-score on local but higher score online... (about 0.005)
def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    score_vail = f1_score(y_true=labels, y_pred=pred, average='macro')
    return '1-f1_score', 1-score_vail   # the goal of XGBoost is to lower target value

## Feature Engineering

def create_feature(df):
    """ Calculate feature for each training sample (a csv with multiple rows) """
    # Initialization
    create_fe = list() # feature data
    col = list()       # feature name

    # First derivative (version 2)
    # d(noncategorical data)/d(活塞工作时长)
    # (some data has only a single row of data, so no differential)
    for i in df.columns:
        if i not in ['活塞工作时长', '设备类型']:
            if i not in ['低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵']:
                if len(df) > 1:
                    first_deri = np.gradient(df[i], df['活塞工作时长'][0])
                else:
                    first_deri = df[i][0]/df['活塞工作时长'][0]
                df[i+'_1st_deri'] = first_deri

    # Creating feature (each one corresponding to exact one label)
    create_fe.append(len(df))
    col.append('data_len')

    # create_fe.append(len(df.drop_duplicates()))
    # col.append('data_drop_dup_len')

    for i in df.columns:
        if i != '设备类型': # i.e. not categorical data

            if i not in ['活塞工作时长', '高压开关', '低压开关']:
                create_fe.append(df[i].max())
                col.append(i+'_max')

                create_fe.append(df[i].min())
                col.append(i+'_min')

                create_fe.append(df[i].std())
                col.append(i+'_std')

            create_fe.append(df[i].sum())
            col.append(i+'_sum')

            create_fe.append(df[i].mean())
            col.append(i+'_mean')

            # create_fe.append(len(df[i].unique()))
            # col.append(i+'_unique_len')
            # create_fe.append(df[i].max()-df[i].min())        
            # col.append(i+'max_min_sub')
            # create_fe.append(df[i].std()/df[i].mean())  
            # col.append(i+'std_mean_sub')
            # create_fe.append(df[i].skew())
            # col.append(i+'_skew')

    total_turn = np.sum(df['活塞工作时长']*df['发动机转速'])
    create_fe.append(total_turn)
    col.append('total_turn')

    return create_fe, col

def get_uid_csv():
    # In this competition, data is seperated in different csv file with a uid label
    train_uid = os.listdir(data_dir + 'data_train/')
    test_uid = os.listdir(data_dir + 'data_test/')
    return train_uid, test_uid

def get_feature():
    """ return training set feature and test set feature """

    train_uid, test_uid = get_uid_csv()

    # Used to map the label to integer (label encoding)
    equipment = {'ZVe44':0, 'ZV573':1, 'ZV63d':2, 
                 'ZVfd4':3, 'ZVa9c':4, 'ZVa78':5, 'ZV252':6}

    try:
        # If feature pickle exist, load from it
        train_feature = load_variable_from_pickle(feature_dir + 'train_feature_' + version + '.pkl')
        test_feature = load_variable_from_pickle(feature_dir + 'test_feature_' + version + '.pkl')
        print("Using previous feature pickle file with version:", version)
    except:
        print("Generating new features...")
        # Training feature
        train_feature = list()
        for filename in tqdm(train_uid):
            raw_data = pd.read_csv(data_dir + 'data_train/' + filename)
            raw_data['设备类型'] = raw_data['设备类型'].map(equipment)
            feature, col  = create_feature(raw_data)
            train_feature.append(feature)
        train_feature = pd.DataFrame(train_feature, columns=col)
        save_variable_to_pickle(train_feature, feature_dir + 'train_feature_' + version + '.pkl')

        # Test feature
        test_feature = list()
        for filename in tqdm(test_uid):
            raw_data = pd.read_csv(data_dir + 'data_test/' + filename)
            raw_data['设备类型'] = raw_data['设备类型'].map(equipment)
            feature, col = create_feature(raw_data)
            test_feature.append(feature)
        test_feature = pd.DataFrame(test_feature, columns=col)
        save_variable_to_pickle(test_feature, feature_dir + 'test_feature_' + version + '.pkl')
    return train_feature, test_feature

def pair_train_label(train_feature):
    
    label = pd.read_csv(data_dir + 'train_labels.csv')
    label.columns = ['uid', 'label']

    train_uid, _ = get_uid_csv()
    train_feature['uid'] = train_uid

    # merge label related with uid
    train_set = train_feature.merge(label, on=['uid'], how='left')

    return train_set

def get_train_test():
    train_feature, test_feature = get_feature()

    train = pair_train_label(train_feature)

    train_data = train.drop(['uid', 'label'], axis=1)
    train_label = train['label']
    test_data = test_feature

    return train_data, train_label, test_data


def KFold(X_train_all, y_train_all, X_test_all, K=5, plot=True):

    skf = StratifiedKFold(n_splits = K, shuffle = True ,random_state=267)

    f1_scores = []
    xgb_test_data = xgb.DMatrix(X_test_all)

    print("Start {}-fold cross validation".format(K))

    for i, (train_index, test_index) in enumerate(skf.split(X_train_all, y_train_all)):
        
        print("\n## Fold", i+1)

        # Split train & validation set
        X_train, X_validation = X_train_all.iloc[train_index,:].copy(), X_train_all.iloc[test_index,:].copy()
        y_train, y_validation = y_train_all.iloc[train_index].copy(), y_train_all.iloc[test_index].copy()

        xgb_train_data = xgb.DMatrix(X_train, y_train)
        xgb_validation_data = xgb.DMatrix(X_validation, y_validation)

        xgb_params = {"objective": 'binary:logistic',
                    "booster" : "gbtree",
                    "eta": 0.05,
                    # "max_depth": 7,
                    # "learning_rate": 0.05,
                    "subsample": 0.85,
                    'eval_metric': 'auc',
                    "colsample_bytree": 0.86,
                    'gpu_id': 0, 
                    "thread": -1,
                    "seed": 666
                    }

        print('Training label ratio (negative/positive) =', np.sum(y_train==0)/np.sum(y_train==1))
        
        # Train model for each fold
        try:
            xgb_model = joblib.load(model_dir + 'xgb_{}.model'.format(i))
            print("Loading pre-train model xgb_{}.model".format(i))
        except:
            watchlist = [(xgb_train_data, 'train'), (xgb_validation_data, 'eval')]
            xgb_model = xgb.train(xgb_params,
                        xgb_train_data,
                        num_boost_round = 1,
                        evals = watchlist, 
                        feval = f1_score_vail, # optimize with f1 score
                        verbose_eval = 200,
                        early_stopping_rounds = 200)
            joblib.dump(xgb_model, model_dir + 'xgb_{}.model'.format(i))
 
        # Predict
        # for validation
        xgb_valid_X = xgb.DMatrix(X_validation) # validation data without label
        pred_validation = xgb_model.predict(xgb_valid_X, ntree_limit=xgb_model.best_ntree_limit)
        pred_validation[pred_validation>=0.5] = 1
        pred_validation[pred_validation<0.5] = 0
        print('Validation label ratio (negative/positive) =', len(pred_validation[pred_validation==0])/len(pred_validation[pred_validation==1]))
        f1_scores.append(f1_score(y_true=y_validation, y_pred=pred_validation, average='macro'))

        # for test
        pred_test = xgb_model.predict(xgb_test_data, ntree_limit=xgb_model.best_ntree_limit)
        print('Test label ratio (negative/positive) =', len(pred_test[pred_test<0.5])/len(pred_test[pred_test>=0.5]))
        # Collect results
        if i == 0:
            cross_validation_pred = np.array(pred_test).reshape(-1, 1)
        else:
            cross_validation_pred = np.hstack((cross_validation_pred, np.array(pred_test).reshape(-1, 1)))

    print("Feature Importance: (",len(xgb_model.get_fscore()), ")\n", xgb_model.get_fscore()) # Get feature importance of each feature
    if plot:
        xgb.plot_importance(xgb_model, max_num_features=25)
        plt.show()

    print("f1 score summary:", f1_scores, "average:", np.mean(f1_scores))

    return cross_validation_pred

def submit_generate(cross_validation_pred):

    # Voting (among each fold)
    submit_label = []
    for record in cross_validation_pred:
        average = np.mean(record)
        if average >= 0.5:
            submit_label.append(1)
        else:
            submit_label.append(0)

    _, test_uid = get_uid_csv()

    submit = pd.DataFrame({'ID': test_uid})
    submit['Label'] = submit_label

    # Make label 1 in front of label 0
    submit = submit.sort_values('Label', ascending=False).reset_index(drop=True)

    submit.to_csv(submit_dir + 'DCIC_XGBoost_submit_' + version + '.csv', index=False)

def main():
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(submit_dir, exist_ok=True)

    # Create feature and Pair label to training set
    X_train_all, y_train_all, X_test_all = get_train_test()
    print('Total features:', len(X_train_all.columns))
    # Train model and make prediction
    cross_validation_pred = KFold(X_train_all, y_train_all, X_test_all, K=5, plot=False)
    # Generate submit file
    submit_generate(cross_validation_pred)

if __name__ == "__main__":
    main()