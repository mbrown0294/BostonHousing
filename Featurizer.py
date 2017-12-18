import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# UNUSED

def replace(df_train, df_test):
    enc = LabelEncoder()
    train_object_cols = df_train.select_dtypes(include=['object'])  # DF Size (1460,43)
    test_object_cols = df_test.select_dtypes(include=['object'])  # DF Size (1459, 43)

    for colName in train_object_cols:
        column_train = df_train[colName].values
        column_test = df_test[colName].values
        cats = []
        # Collect unique list of categories -----> Includes NaN
        for category in column_train:
            if category in cats:
                continue
            cats.append(category)
        print(cats)                                         # RL, RM, C, FV, RH
        # Replace category value with integer
        print(np.unique(column_train, return_counts=True))  # ['C (all)', 'FV', 'RH', 'RL', 'RM'], dtype=object), array([  10,   65,   16, 1151,  218]
        for ind in range(len(column_train)):
            for category in cats:
                if category == column_train[ind]:
                    column_train[ind] = cats.index(category)
        # print(column_train)
        break


        # for val in column_test:
        #     for category in cats:
        #         if category == column_test[val]:
        #             column_test[val] = cats.index(category)

        # Replaced NaN values with NaN category index
        break
        # df_train[colName].fillna(len(cats)-1)

    df_train_objects = df_train.select_dtypes(include=[object])
    df_test_objects = df_test.select_dtypes(include=[object])

    df_train_objects = df_train_objects.fillna('NA')
    df_test_objects = df_test_objects.fillna('NA')

    # print(df_train['Street'].value_counts())
    # print(df_test['Street'].value_counts())

    encoded_df_train = pd.DataFrame(index=df_train_objects.index)
    encoded_df_test = pd.DataFrame(index=df_test_objects.index)

    encoded_df_train[column] = enc.fit_transform(df_train_objects[column])
    encoded_df_test[column] = enc.transform(df_test_objects[column])

    # encoded_df_test = df_test_objects.apply(enc.transform)
    #
    # encoded_df_train = encoded_df_train.applymap(str)
    # encoded_df_test = encoded_df_test.applymap(str)
    #
    # dummy_df_train = pd.get_dummies(encoded_df_train, drop_first=True)
    # dummy_df_train.to_csv('dummyTrain.csv')
    #
    # dummy_df_test = pd.get_dummies(encoded_df_test, drop_first=True)
    # dummy_df_test.to_csv('dummyTest.csv')
    #
    # df_train = df_train.drop(df_train_objects, 1)
    # df_train = df_train.join(dummy_df_train)
    #
    # df_test = df_test.drop(df_test_objects, 1)
    # df_test = df_test.join(dummy_df_test)
    #
    # return df_train, df_test


def featurize(df_train, df_test):
    trainer, tester = replace(df_train, df_test)
    trainer.to_csv("featurizedTrain.csv", index=False)
    tester.to_csv("featurizedTest.csv", index=True)
    return trainer, tester


def run(df_train, df_test):
    if 'Id' in df_train.columns:
        df_train.set_index('Id', drop=True, inplace=True)
    if 'Id' in df_test.columns:
        df_test.set_index('Id', drop=True, inplace=True)
    data_train, data_test = featurize(df_train, df_test)
    # print(data_train)
    #print(data_test)


if __name__ == '__main__':
    train = pd.read_csv('cleanTrain.csv')
    test = pd.read_csv('cleanTest.csv')
    run(train, test)
    # run(test, encoder, False)
