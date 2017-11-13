from sklearn import linear_model
import pandas as pd


def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def predict(model, x_test):
    predicted_price = model.predict(x_test)
    return predicted_price


def evaluate(y_true, y_pred, metric):
    return metric(y_true, y_pred)


if __name__ == "__main__":
    housing_train = pd.read_csv("featurizedTrain.csv")
    # Shape: (1460, 79)
    housing_test = pd.read_csv("featurizedTest.csv")
    housing_test.set_index('Id', drop=True, inplace=True)  # Maintains 'Id' values
    # Shape: (1459, 79)
    # Setting the models
    logReg = linear_model.LogisticRegression()
    linReg = linear_model.LinearRegression()
    # Creating the CSV
    train_x = housing_train.values
    train_y = pd.read_csv("train_prices.csv").values
    test_x = housing_test.values
    train(linReg, x_train=train_x, y_train=train_y)  # Trains model (works)
    index = housing_test.index
    price_pred = predict(linReg, housing_test)  # Predicted prices (got values, at least)
    pred_df = pd.DataFrame(index=index)
    pred_df['SalePrice'] = price_pred

    # print(pred_df)
    # Getting negatives for some reason. Might be incorrectly featurized?
    # Following for-loop returns all negatives predicted and their indices

    # for i in range(len(pred_df.SalePrice)):
    #     if pred_df.get_value(i, 0, takeable=True) < 0:
    #         print(i, ", ", pred_df.get_value(i, 0, takeable=True))

    pred_df.to_csv("Submission.csv")
