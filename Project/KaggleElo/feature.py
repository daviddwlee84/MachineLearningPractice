import pandas as pd
import numpy as np
from datetime import datetime, date

data_path = 'data/'
deal_path = 'deal/'
feat_path = 'feat/'

def init_data():
    """ Transfer csv format into pickle format """
    data_train = pd.read_csv('raw_data/train.csv')
    data_train.to_pickle(data_path + 'train.pickle')
    del data_train

    data_test = pd.read_csv('raw_data/test.csv')
    data_test.to_pickle(data_path + 'test.pickle')
    del data_test

    # The file size is too large that on OSX will cause
    # OSError: [Errno 22] Invalid argument
    # in pandas to_pickle, the pickle.dump()
    # data_history = pd.read_csv('raw_data/historical_transactions.csv')
    # data_history.to_pickle(data_path + 'historical_transactions.pickle')
    # del data_history

    data_merchants = pd.read_csv('raw_data/merchants.csv')
    data_merchants.to_pickle(data_path + 'merchants.pickle')
    del data_merchants

    data_new_merchant_transactions = pd.read_csv('raw_data/new_merchant_transactions.csv')
    data_new_merchant_transactions.to_pickle(data_path + 'new_merchant_transactions.pickle')
    del data_new_merchant_transactions

def deal_data(): # reference from Elo World
    """ Some data preprocessing """
    def read_train_test_data(pickle_file_name):
        df = pd.read_pickle(data_path + pickle_file_name)
        
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['elapsed_time'] = (date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

        # ValueError: DataFrame.dtypes for data must be int, float or bool.
        # Did not expect the data types in fields first_active_month
        # Name: first_active_month, dtype: datetime64[ns]
        df = df.drop('first_active_month', axis=1)

        return df

    train = read_train_test_data('train.pickle')
    test = read_train_test_data('test.pickle')

    train.to_pickle(deal_path + 'train.pickle')
    test.to_pickle(deal_path + 'test.pickle')

    def reduce_mem_usage(df, verbose=True): # reference from Elo World
        """ Helper function to reduce memory usage """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

        return df

    # Parse dates from new_merchant_transactions and historical_transactions
    # to load from csv we can use pd.read_csv('filepath', parse_dates=['purchase_date'])
    # here we do the same thing in two different demonstration
    deal_historical_transactions = pd.read_csv('raw_data/historical_transactions.csv', parse_dates=['purchase_date']) # read from csv caused by OSError of pickle
    deal_new_merchant_transactions = pd.read_pickle(data_path + 'new_merchant_transactions.pickle')
    deal_new_merchant_transactions['purchase_date'] = pd.to_datetime(deal_new_merchant_transactions['purchase_date'])
    
    # Binarize some columns from Y/N to 1/0
    # (make booleans features to numeric)
    cols_to_binarize = ['authorized_flag', 'category_1']
    for col in cols_to_binarize:
        deal_historical_transactions[col] = deal_historical_transactions[col].map({'Y':1, 'N':0})
        deal_new_merchant_transactions[col] = deal_new_merchant_transactions[col].map({'Y':1, 'N':0})

    ## Feature Engineering
    deal_historical_transactions['month_diff'] = ((datetime.today() - deal_historical_transactions['purchase_date']).dt.days)//30
    deal_historical_transactions['month_diff'] += deal_historical_transactions['month_lag']
    deal_new_merchant_transactions['month_diff'] = ((datetime.today() - deal_new_merchant_transactions['purchase_date']).dt.days)//30
    deal_new_merchant_transactions['month_diff'] += deal_new_merchant_transactions['month_lag']

    # Convert categorical data into dummy/indicator variables
    deal_historical_transactions = pd.get_dummies(deal_historical_transactions, columns=['category_2', 'category_3'])
    deal_new_merchant_transactions = pd.get_dummies(deal_new_merchant_transactions, columns=['category_2', 'category_3'])

    # Reduce memory usage
    deal_historical_transactions = reduce_mem_usage(deal_historical_transactions)
    deal_new_merchant_transactions = reduce_mem_usage(deal_new_merchant_transactions)

    # Aggregate authorized mean
    agg_fun = {'authorized_flag': ['mean']}
    feat_authorized_mean = deal_historical_transactions.groupby(['card_id']).agg(agg_fun) # get mean value for each card_id
    feat_authorized_mean.columns = ['_'.join(col).strip() for col in feat_authorized_mean.columns.values]
    feat_authorized_mean.reset_index(inplace=True)
    feat_authorized_mean.to_pickle(feat_path + 'authorized_mean.pickle') # output this feature to pickle


    authorized_transactions = deal_historical_transactions[deal_historical_transactions['authorized_flag'] == 1]
    deal_historical_transactions = deal_historical_transactions[deal_historical_transactions['authorized_flag'] == 0]
    
    authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
    deal_historical_transactions['purchase_month'] = deal_historical_transactions['purchase_date'].dt.month
    deal_new_merchant_transactions['purchase_month'] = deal_new_merchant_transactions['purchase_date'].dt.month

    # save files into pickle format
    authorized_transactions.to_pickle(deal_path + 'authorized_transactions.pickle')
    deal_historical_transactions.to_pickle(deal_path + 'historical_transactions.pickle')
    deal_new_merchant_transactions.to_pickle(deal_path + 'new_merchant_transactions.pickle')

def feat_data():
    """ Building features """
    # Aggregate functions
    # aggregate the info contained in
    # authorized_transactions, historical_transactions
    # and new_merchant_transactions
    def aggregate_transactions(history): # reference from Elo World
        """ Aggregates the function by grouping on card_id """

        history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                        astype(np.int64) * 1e-9
        
        agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['mean', 'max', 'min', 'std'],
        'month_diff': ['mean']
        }
        
        agg_history = history.groupby(['card_id']).agg(agg_func)
        agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
        agg_history.reset_index(inplace=True)
        
        df = (history.groupby('card_id')
            .size()
            .reset_index(name='transactions_count'))
        
        agg_history = pd.merge(df, agg_history, on='card_id', how='left')
        
        return agg_history

    historical_transactions = pd.read_pickle(deal_path + 'historical_transactions.pickle')
    feat_historical_transactions = aggregate_transactions(historical_transactions)
    feat_historical_transactions.columns = ['hist_' + col if col != 'card_id' else col for col in feat_historical_transactions.columns]
    feat_historical_transactions.to_pickle(feat_path + 'historical_transactions.pickle')

    authorized_transactions = pd.read_pickle(deal_path + 'authorized_transactions.pickle')
    feat_authorized_transactions = aggregate_transactions(authorized_transactions)
    feat_authorized_transactions.columns = ['auth_' + col if col != 'card_id' else col for col in feat_authorized_transactions.columns]
    feat_authorized_transactions.to_pickle(feat_path + 'authorized_transactions.pickle')

    new_merchant_transactions = pd.read_pickle(deal_path + 'new_merchant_transactions.pickle')
    feat_new_transactions = aggregate_transactions(new_merchant_transactions)
    feat_new_transactions.columns = ['new_' + col if col != 'card_id' else col for col in feat_new_transactions.columns]
    feat_new_transactions.to_pickle(feat_path + 'new_merchant_transactions.pickle')

    def aggregate_per_month(history):
        """
        1. Aggregate on the two variables card_id and month_lag.
        2. Aggregate over time
        """

        grouped = history.groupby(['card_id', 'month_lag'])

        agg_func = {
                'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
                'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
                }

        intermediate_group = grouped.agg(agg_func)
        intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
        intermediate_group.reset_index(inplace=True)

        final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
        final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
        final_group.reset_index(inplace=True)
        
        return final_group

    feat_final_group = aggregate_per_month(authorized_transactions)
    feat_final_group.to_pickle(feat_path + 'final_group.pickle')

    def successive_aggregates(df, field1, field2):
        t = df.groupby(['card_id', field1])[field2].mean()
        u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
        u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
        u.reset_index(inplace=True)
        return u

    feat_additional_fields = successive_aggregates(new_merchant_transactions, 'category_1', 'purchase_amount')
    feat_additional_fields = feat_additional_fields.merge(successive_aggregates(new_merchant_transactions, 'installments', 'purchase_amount'),
                                            on = 'card_id', how='left')
    feat_additional_fields = feat_additional_fields.merge(successive_aggregates(new_merchant_transactions, 'city_id', 'purchase_amount'),
                                            on = 'card_id', how='left')
    feat_additional_fields = feat_additional_fields.merge(successive_aggregates(new_merchant_transactions, 'category_1', 'installments'),
                                            on = 'card_id', how='left')
    feat_additional_fields.to_pickle(feat_path + 'additional_fields.pickle')

if __name__=='__main__':
    print("Reading data from csv raw file into pickle file format")
    init_data() # Reading data from csv raw file into pickle file format

    print("Do some data preprocessing to deal with data")
    deal_data() # Do some data preprocessing to deal with data

    print("Making features")
    feat_data() # Make features
