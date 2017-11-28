import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode(train_df, test_df, enc):
    # Creates subset of DataFrames with only object columns
    train_obj = train_df.select_dtypes(include=['object'])
    test_obj = test_df.select_dtypes(include=['object'])

    col_names = train_obj.columns

    # Sums up the nulls in each column
    train_nulls = train_obj.isnull().sum()
    test_nulls = test_obj.isnull().sum()
    # print(train_nulls, '\n\nDone\n\n', test_nulls)

    # Creates and transposes matrices to represent DataFrames
    train = train_obj.values
    train = np.matrix.transpose(train)
    test = test_obj.values
    test = np.matrix.transpose(test)

    # print(train, '\n\nNEXT\n\n', test)

    for x in range(len(train)):
        # print(np.unique(train[x].astype(str), return_counts=True))  # Essentially value_counts for an array
        train[x] = enc.fit_transform(train[x].astype(str))
        if x <= (len(test)-1):
            test[x] = enc.transform(test[x].astype(str))

    # print('\n\n\nDONE\n\n\n', train, '\n\nNEXT\n\n', test)
    return train, test, col_names


def re_dataframe(train_mat, test_mat):
    train_mat = np.matrix.transpose(train_mat)
    test_mat = np.matrix.transpose(test_mat)
    new_train_df = pd.DataFrame(train_mat)
    new_test_df = pd.DataFrame(test_mat)
    new_train_df.columns = columns
    new_test_df.columns = columns
    new_test_df.set_index(index_test, inplace=True)
    # print(new_train_df, '\n\n\nHI\n\n\n', new_test_df)
    return new_train_df, new_test_df


def one_hot(train_df, test_df):
    dummy_train = pd.get_dummies(train_df, drop_first=True)
    dummy_test = pd.get_dummies(test_df, drop_first=True)
    # print(dummy_train, '\n\n\nNEW\n\n\n', dummy_test)
    return dummy_train, dummy_test


def rejoin(full_train, full_test, train_dum, test_dum):
    full_train.drop(full_train.select_dtypes(include=['object']), 1, inplace=True)
    full_test.drop(full_test.select_dtypes(include=['object']), 1, inplace=True)

    # print(full_train, '\n\n\n\nHI\n\n\n\n', full_test)

    full_train = full_train.join(train_dum)
    full_test = full_test.join(test_dum)

    # print('\n\n\n\n\n\n\nAHHHHHHHHHHHHHHHHHHHHHHHHHH\n\n\n\n\n\n\n', full_train, '\n\n\n\nHI\n\n\n\n', full_test)

    return full_train, full_test


if __name__ == '__main__':
    encoder = LabelEncoder()

    train_dataframe = pd.read_csv("cleanTrain.csv")
    if 'Id' in train_dataframe.columns:
        train_dataframe.set_index('Id', drop=True, inplace=True)
    index_train = train_dataframe.index

    test_dataframe = pd.read_csv("cleanTest.csv")
    if 'Id' in test_dataframe.columns:
        test_dataframe.set_index('Id', drop=True, inplace=True)
    index_test = test_dataframe.index

    # print(index_train, '\n', index_test)

    train_dataframe.drop(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional',
                          'SaleType'], 1, inplace=True)
    test_dataframe.drop(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional',
                         'SaleType'], 1, inplace=True)

    encoded_train, encoded_test, columns = encode(train_dataframe, test_dataframe, encoder)
    new_train, new_test = re_dataframe(encoded_train, encoded_test)
    train_dummy, test_dummy = one_hot(new_train, new_test)
    train_dataframe, test_dataframe = rejoin(train_dataframe, test_dataframe, train_dummy, test_dummy)
    # print(train_dataframe, '\n\n\n\n\n\n\n\n\n\n\n\n\n', test_dataframe)
    train_dataframe.to_csv('featurized_train.csv', index=False)
    test_dataframe.to_csv('featurized_test.csv', index=True)
