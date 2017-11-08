import numpy as np
import pandas as pd


def number_clean(df):
    is_numeric = ((df.dtypes == "int64") | (df.dtypes == "float64")).values
    numeric_columns = df.columns[is_numeric]
    for col_name in numeric_columns:
        column = df[col_name].values
        median = np.nanmedian(column)
        df.at[np.isnan(column), col_name] = median


def object_clean(df):
    df.CentralAir = np.where(df.CentralAir == 'Y', 1, 0)
    df.PavedDrive = np.where(df.PavedDrive == 'Y', 1, 0)


def clean(df, is_train):
    object_clean(df)
    number_clean(df)
    if is_train:
        df.to_csv("cleanTrain.csv", index=True)
    else:
        df.to_csv("cleanTest.csv", index=True)
    return df


def run(csv, is_train):
    housing = pd.read_csv(csv)
    housing.set_index('Id', drop=True, inplace=True)
    if 'SalePrice' in housing.columns:
        housing.SalePrice.to_csv('train_prices.csv')
        housing.drop('SalePrice', 1, inplace=True)
    data = clean(housing, is_train)
    print(data)


if __name__ == '__main__':
    run('train.csv', True)
    run('test.csv', False)

