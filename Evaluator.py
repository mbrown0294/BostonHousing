from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error
# from sklearn.feature_selection import SelectFromModel


def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def predict(model, x_test):
    price_pred = model.predict(x_test)
    return price_pred


def evaluate(y_true, y_pred, metric):
    # selector = SelectFromModel(model, )
    return metric(y_true, y_pred)


if __name__ == "__main__":
    housing_train = pd.read_csv("featurizedTrain.csv")
    housing_test = pd.read_csv("featurizedTest.csv")
    housing_test.set_index('Id', drop=True, inplace=True)
    # prices = housing.SalePrice.values
    # values = housing.drop("SalePrice", 1).values
    # Setting the models
    logReg = linear_model.LogisticRegression()
    linReg = linear_model.LinearRegression()
    # Creating the CSV
    x_train = housing_train.drop('SalePrice',1).values
    y_train = housing_train.SalePrice.values
    x_test = housing_test.values
    train(linReg, x_train=x_train, y_train=y_train)
    index = housing_test.index
    pred_df = pd.DataFrame(index=index)
    price_pred = predict(linReg,housing_test)
    pred_df['SalePrice'] = price_pred
    pred_df.to_csv("Submission.csv")

    # print(evaluate(values, prices, logReg, mean_squared_error))  # 91950248.2877
    # print(evaluate(values, prices, linReg, mean_squared_error))  # 867849985.001
