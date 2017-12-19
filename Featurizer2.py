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

# # # THERE'S A NAN IN MY CODE # # #

# Turns matrices back into DataFrames and adds index for test
def re_dataframe(train_mat, test_mat):
    train_mat = np.matrix.transpose(train_mat)
    test_mat = np.matrix.transpose(test_mat)
    new_train_df = pd.DataFrame(train_mat, index=index_train)
    new_test_df = pd.DataFrame(test_mat, index=index_test)
    new_train_df.columns = columns
    new_test_df.columns = columns
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


if __name__ == '__main__':
    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Creates DataFrames and preserves index
    train_dataframe = pd.read_csv("cleanTrain.csv")  # (1460, 72)
    if 'Id' in train_dataframe.columns:
        train_dataframe.set_index('Id', drop=True, inplace=True)
    index_train = train_dataframe.index
    test_dataframe = pd.read_csv("cleanTest.csv")  # (1459, 72)
    if 'Id' in test_dataframe.columns:
        test_dataframe.set_index('Id', drop=True, inplace=True)
    index_test = test_dataframe.index

    # Drops columns with categories in Test but not Train
    train_dataframe.drop(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
                          ], 1, inplace=True)
    test_dataframe.drop(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
                         ], 1, inplace=True)

    # Variables
    X = train_dataframe.values
    y = pd.read_csv("train_prices.csv")
    y = y.values

    # Separate object from number
    train_obj = train_dataframe.select_dtypes(include=['object'])
    train_num = train_dataframe.drop(train_obj, 1)  # (1460, 36)
    test_obj = test_dataframe.select_dtypes(include=['object'])
    test_num = test_dataframe.drop(test_obj, 1)  # (1459, 36)
    train_cols = train_num.columns
    test_cols = test_num.columns

    # Start scaling (to between 1 and 0)

    scaled_train = scaler.fit_transform(train_num)  # (1460, 36)
    scaled_test = scaler.transform(test_num)  # (1459, 36)
    train = pd.DataFrame(scaled_train, columns=train_cols)
    test = pd.DataFrame(scaled_test, columns=test_cols, index=index_test)
    train_dataframe = train.join(train_obj)
    test_dataframe = test.join(test_obj)

    # Label encodes category columns
    encoded_train, encoded_test, columns = encode(train_dataframe, test_dataframe, encoder)

    # Turn encoded matrix back into DataFrame
    new_train, new_test = re_dataframe(encoded_train, encoded_test)

    # One-hot encodes
    train_dummy, test_dummy = one_hot(new_train, new_test)  # Train: (1460, 194) / Test: (1459, 159)

    # print(type(train_dataframe))
    # print(train_dataframe.dtypes)

    # Rejons one-hot with numeric
    train_dataframe, test_dataframe = rejoin(train_dataframe, test_dataframe, train_dummy, test_dummy)
    # train_dataframe.to_csv("temp.csv")

    # # Finishes with a CSV
    # train_dataframe.to_csv('featurized_train.csv', index=True)
    # test_dataframe.to_csv('featurized_test.csv', index=True)
