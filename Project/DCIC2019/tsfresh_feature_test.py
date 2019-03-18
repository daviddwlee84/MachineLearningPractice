import pandas as pd
import numpy as np
import os
from tqdm import tqdm

data_dir = 'raw_data/'
output_dir = 'combined_data/' # todo mkdir
output_file = 'train.csv'

def get_train_uid_and_label():
    label = pd.read_csv(data_dir + 'train_labels.csv')
    label.columns = ['uid', 'label']
    return list(label['uid']), label['label']

def combine_data(train_uid):
    for id, uid in tqdm(enumerate(train_uid)):
        filename = data_dir + 'data_train/' + uid
        temp_train = pd.read_csv(filename)
        temp_train = temp_train[['发动机转速', '油泵转速', '泵送压力', '液压油温', '流量档位', '分配压力', '排量电流']]
        # temp_train['uid_csv'] = str(uid)
        temp_train['id'] = id
        length = temp_train.shape[0]
        temp_train['time'] = [i for i in range(length)]
        if id == 0:
            temp_train.to_csv(output_dir + output_file, index=False)
        else:
            temp_train.to_csv(output_dir + output_file, index=False, header=False, mode='a+')

def find_feature(timeseries, y):
    from tsfresh import extract_relevant_features
    features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
    return features_filtered_direct

def main():
    train_uid, label = get_train_uid_and_label()
    if not os.path.isfile(output_dir + output_file):
        print("Combining training data...")
        combine_data(train_uid)
    else:
        print("Found combined training data!")
    
    train_data = pd.read_csv(output_dir + output_file)

    print("Finding feature....")
    feature = find_feature(train_data, label)
    print(feature)

    feature.to_csv('tsfresh_feat.csv')

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    main()
