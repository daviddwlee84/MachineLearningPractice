from scipy.io.mmio import mmread
import os.path
import random # for sample selection

data_path = "Datasets/Epinions"
train_name = "EP25_UPL5_train.mtx"
test_name = "EP25_UPL5_test.mtx"

def load_epinions(path=data_path):
    # Scipy CSR: Compressed Sparce Row format
    train_data = mmread(os.path.join(path, train_name)).tocsr()
    test_data = mmread(os.path.join(path, test_name)).tocsr()

    return train_data, test_data

def get_sample_users(train_data, test_data=None, max_user_sample=1000):
    num_train_sample_users = min(train_data.shape[0], max_user_sample)
    train_sample_users = random.sample(range(train_data.shape[0]), num_train_sample_users)
    if test_data is not None:
        num_test_sample_users = min(test_data.shape[0], max_user_sample)
        test_sample_users = random.sample(range(test_data.shape[0]), num_test_sample_users)
        return train_sample_users, test_sample_users
    else:
        return train_sample_users
