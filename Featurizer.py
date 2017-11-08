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
    new_df = (pd.get_dummies(df_objects))
    new_df['SalePrice'] = y
    return df


def featurize(df):
    replace(df).to_csv("featurizedTrain.csv", index=False)
    new_housing_csv = pd.read_csv("featurizedTrain.csv")
    return new_housing_csv


if __name__ == '__main__':
    housing = pd.read_csv("cleanTrain.csv")
    raw2 = pd.read_csv("cleanTrain.csv")
    y = housing.SalePrice.values
    print(featurize(housing))

