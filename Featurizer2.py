import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode(train_df, test_df, enc):
    # Creates subset of DataFrames with only object columns
    train_object = train_df.select_dtypes(include=['object'])  # Index starts at 0
    test_object = test_df.select_dtypes(include=['object'])
    col_names = train_object.columns

    # Creates and transposes matrices to represent DataFrames
    train_vals = train_object.values
    train_vals = np.matrix.transpose(train_vals)
    test_vals = test_object.values
    test_vals = np.matrix.transpose(test_vals)
    # LabelEncodes both matrices
    for x in range(len(train_vals)):
        # print(np.unique(train[x].astype(str), return_counts=True))  # Essentially value_counts for an array
        train_vals[x] = enc.fit_transform(train_vals[x].astype(str))
        if x <= (len(test_vals)-1):
            test_vals[x] = enc.transform(test_vals[x].astype(str))
    return train_vals, test_vals, col_names


# Turns matrices back into DataFrames and adds index for test
def re_dataframe(train_mat, test_mat):
    train_mat = np.matrix.transpose(train_mat)
    test_mat = np.matrix.transpose(test_mat)
    new_train_df = pd.DataFrame(train_mat, index=ind_train)
    new_test_df = pd.DataFrame(test_mat, index=ind_test)
    new_train_df.columns = cols
    new_test_df.columns = cols
    return new_train_df, new_test_df


# One-hot encodes LabelEncoded dataframes
def one_hot(train_df, test_df):
    dummy_train = pd.get_dummies(train_df, drop_first=True)
    dummy_test = pd.get_dummies(test_df, drop_first=True)
    return dummy_train, dummy_test


# Joins numerical subsets of DataFrames with one-hot encoded object subsets
def rejoin(full_train, full_test, train_dum, test_dum):
    # Drops object columns
    full_train.drop(full_train.select_dtypes(include=['object']), 1, inplace=True)
    full_test.drop(full_test.select_dtypes(include=['object']), 1, inplace=True)

    # Joins with numeric
    full_train = full_train.join(train_dum)
    full_test = full_test.join(test_dum)
    return full_train, full_test


def scale_func(tr, te):
    scaled_tr = scaler.fit_transform(tr)  # (1460, 36)
    scaled_te = scaler.transform(te)  # (1459, 36)
    scaled_tr_num = pd.DataFrame(scaled_tr, columns=num_columns, index=ind_train)
    scaled_te_num = pd.DataFrame(scaled_te, columns=num_columns, index=ind_test)
    return scaled_tr_num, scaled_te_num


if __name__ == '__main__':
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    scale = False

    # Creates DataFrames and preserves index
    train = pd.read_csv("clean_train.csv")  # (1460, 72)
    if 'Id' in train.columns:
        train.set_index('Id', drop=True, inplace=True)
    ind_train = train.index  # (1, 1460)
    test = pd.read_csv("clean_test.csv")  # (1459, 72)
    if 'Id' in test.columns:
        test.set_index('Id', drop=True, inplace=True)
    ind_test = test.index

    # Drops columns with categories in Test but not Train
    dropped = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
    train, test = train.drop(dropped, 1, inplace=False), test.drop(dropped, 1, inplace=False)
    # Variables
    X = train.values
    y = pd.read_csv("ground_truth.csv")
    y = np.squeeze(y.values)
    columns = train.columns

    # Separate object from number <------ No NaNs
    obj_train = train.select_dtypes(include=['object'])
    obj_test = test.select_dtypes(include=['object'])
    num_train = train.drop(obj_train, 1)  # (1460, 36)
    num_test = test.drop(obj_test, 1)  # (1459, 36)
    num_columns = num_train.columns
    obj_columns = obj_train.columns

    if scale:
        scaled_train, scaled_test = scale_func(num_train, num_test)
        train = scaled_train.join(obj_train)
        test = scaled_test.join(obj_test)

    # Label encodes category columns
    encoded_train, encoded_test, cols = encode(obj_train, obj_test, encoder)

    # Turn encoded matrix back into DataFrame
    new_train, new_test = re_dataframe(encoded_train, encoded_test)

    # One-hot encodes
    train_dummy, test_dummy = one_hot(obj_train, obj_test)  # Train: (1460, 194) / Test: (1459, 159)
    columns = train_dummy.columns
    # print(list(set(columns) - set(test_dummy.columns)))  # <--- Shows difference in Series.
    # More columns in Train set. All test are in train

    # Rejons one-hot with numeric
    train, test = rejoin(train, test, train_dummy, test_dummy)
    print(train, test)
    # Finishes with a CSV
    train.to_csv('new_feat_train.csv', index=True)
    test.to_csv('new_feat_test.csv', index=True)
