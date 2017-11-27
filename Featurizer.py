import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def replace(df_train, df_test):
    enc = LabelEncoder()
    # object_column_names = df.columns[is_object]
    # for colName in object_column_names:
    #     column = df[colName].values
    #     # cats = []
    #     # # Collect unique list of categories
    #     # for category in column:
    #     #     if category in cats:
    #     #         continue
    #     #     cats.append(category)
    #     # # Replace category value with integer
    #     # print(cats)
    #     # cat_per_col
    #     for index in column:
    #         # for category in cats:
    #         #     if category == column[index]:
    #         #         column[index] = cats.index(category)
    #     # Replaced NaN values with NaN category index
    #         if ((type(index) == float and type(column[index]) == float) and

    df_test.drop('MSZoning', 1, inplace=True)
    df_train.drop('MSZoning', 1, inplace=True)
    df_train_objects = df_train.select_dtypes(include=[object])
    df_test_objects = df_test.select_dtypes(include=[object])

    df_train_objects = df_train_objects.fillna('NaN')
    df_test_objects = df_test_objects.fillna('NaN')

    # print(df_train['Street'].value_counts())
    # print(df_test['Street'].value_counts())

    encoded_df_train = pd.DataFrame(index=df_train_objects.index)
    encoded_df_test = pd.DataFrame(index=df_test_objects.index)
    for column in df_train_objects.columns:
        encoded_df_train[column] = df_train_objects[column].apply(enc.fit_transform)
        encoded_df_test[column] = df_test_objects[column].apply(enc.transform)

    encoded_df_test = df_test_objects.apply(enc.transform)

    encoded_df_train = encoded_df_train.applymap(str)
    encoded_df_test = encoded_df_test.applymap(str)

    dummy_df_train = pd.get_dummies(encoded_df_train, drop_first=True)
    dummy_df_train.to_csv('dummyTrain.csv')

    dummy_df_test = pd.get_dummies(encoded_df_test, drop_first=True)
    dummy_df_test.to_csv('dummyTest.csv')

    df_train = df_train.drop(df_train_objects, 1)
    df_train = df_train.join(dummy_df_train)

    df_test = df_test.drop(df_test_objects, 1)
    df_test = df_test.join(dummy_df_test)

    return df_train, df_test


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
