import numpy as np
import pandas as pd


def number_clean(df):
    # df.drop("SalePrice", 1, inplace=True)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    is_numeric = ((df.dtypes == "int64") | (df.dtypes == "float64")).values
    numeric_columns = df.columns[is_numeric]
    for col_name in numeric_columns:
        column = df[col_name].values
        median = np.nanmedian(column)
        df.at[np.isnan(column), col_name] = median


def object_clean(df):
    df.CentralAir = np.where(df.CentralAir == 'Y', 1, 0)
    df.PavedDrive = np.where(df.PavedDrive == 'Y', 1, 0)


def clean(df, og):
    object_clean(df)
    number_clean(df)
    df[og] = y
    df.to_csv("cleanTrain.csv", index=False)
    return df


if __name__ == '__main__':
    housing = pd.read_csv("train.csv")
    raw1 = pd.read_csv("train.csv")
    y = housing.SalePrice.values
    y_name = 'SalePrice'
    print(clean(housing, y_name))
