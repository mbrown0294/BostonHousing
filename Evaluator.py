from sklearn import linear_model
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier


def train(selector, model, x_train, y_train, x_test):
    selector = selector(chi2, k=12)
    x_train_clean = selector.fit_transform(x_train, y_train)
    x_test_clean = selector.transform(x_test)
    model.fit(x_train_clean, y_train)
    return x_test_clean


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
    new_test_x = train(SelectKBest, linReg, train_x, train_y, test_x)  # Trains model
    index = housing_test.index
    price_pred = predict(linReg, new_test_x)  # Predicted prices (got values, at least)
    pred_df = pd.DataFrame(index=index)
    pred_df['SalePrice'] = price_pred
    pred_df.to_csv("Submission.csv")

# Getting negatives for some reason. Might be incorrectly featurized?
    # Following for-loop returns all negatives predicted and their indices
    #
    for i in range(len(pred_df.SalePrice)):
        if pred_df.get_value(i, 0, takeable=True) < 0:
            print(i, ", ", pred_df.get_value(i, 0, takeable=True))
