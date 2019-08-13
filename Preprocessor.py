import numpy as np
import pandas as pd


def number_clean(df):
    is_numeric = ((df.dtypes == "int64") | (df.dtypes == "float64")).values
    numeric_columns = df.columns[is_numeric]
    for col_name in numeric_columns:
        column = df[col_name].values
        median = np.nanmedian(column)
        df.at[np.isnan(column), col_name] = median


def clean(df, is_train):
    number_clean(df)
    if is_train:
        df.to_csv("clean_train.csv", index=True)  # Index is reset each new csv
    else:
        df.to_csv("clean_test.csv", index=True)  # Want to keep index to keep order
    return df


def run(csv, is_train):
    housing = pd.read_csv(csv)
    housing.set_index('Id', drop=True, inplace=True)
    if 'SalePrice' in housing.columns:
        prices = housing.SalePrice.values
        prices_df = pd.DataFrame({"SalePrice": prices})  # Change 'prices' into a df with column name and everything
        prices_df.to_csv('ground_truth.csv', index=False)
        housing.drop('SalePrice', 1, inplace=True)
    data = clean(housing, is_train)


if __name__ == '__main__':
    run('train.csv', True)
    run('test.csv', False)
