import numpy as np
import pandas as pd


def replace(df):
    is_object = (df.dtypes == "object").values
    object_column_names = df.columns[is_object]
    for colName in object_column_names:
        column = df[colName].values
        cats = []
        # Collect unique list of categories
        for category in column:
            if category in cats:
                continue
            cats.append(category)
        # Replace category value with integer
        for index in range(len(column)):
            for category in cats:
                if category == column[index]:
                    column[index] = cats.index(category)
                # Replaced NaN values with NaN category index
                if ((type(category) == float and type(column[index]) == float) and
                        (np.isnan(category) and np.isnan(column[index]))):
                    column[index] = cats.index(category)
    df_objects = df.select_dtypes(include=[object])
    dummy_df = pd.get_dummies(df_objects, drop_first=True)
    dummy_df.to_csv('dummies.csv')
    return df


def featurize(df, is_train):
    replace(df)
    if is_train:
        df.to_csv("featurizedTrain.csv", index=False)
    else:
        df.to_csv("featurizedTest.csv", index=True)
    return df


def run(csv, is_train):
    housing = pd.read_csv(csv)
    if not is_train:
        housing.set_index('Id', drop=True, inplace=True)
    data = featurize(housing, is_train)
    print(data)


if __name__ == '__main__':
    run('cleanTrain.csv', True)
    run('cleanTest.csv', False)
