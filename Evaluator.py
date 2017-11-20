import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer , mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# from sklearn.feature_selection import SelectKBest


# Trains the data with a SelectKBest model
def train(selector, mod, train_x, train_y, x_test):
    selector = selector(chi2, k=12)
    x_train_clean = selector.fit_transform(train_x, train_y)
    x_test_clean = selector.transform(x_test)
    mod.fit(x_train_clean, train_y)
    return x_test_clean


# Creates predicted values of 'y'
def pred(mod, x_test):
    predicted_price = mod.predict(x_test)
    return predicted_price


def evaluate(y_true, y_pred, metr):
    return metr(y_true, y_pred)


def grid_search(trainx, trainy):
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]
    clf = GridSearchCV(estimator=svm.SVR(), param_grid=parameter_candidates, n_jobs=-1)
    trainy = np.squeeze(trainy)
    clf.fit(trainx, trainy)
    print(clf.best_score_)


if __name__ == "__main__":
    housing_train = pd.read_csv("featurizedTrain.csv")
    housing_test = pd.read_csv("featurizedTest.csv")
    housing_test.set_index('Id', drop=True, inplace=True)  # Maintains 'Id' values
# Set metric
    # metric = mean_squared_error
    # metric = median_absolute_error
    # metric = mean_absolute_error
    metric = mean_squared_log_error
    # metric = r2_score
    # metric = explained_variance_score
# Set model
    # model = LinearRegression()
    model = RandomForestRegressor()
    # model = LogisticRegression()
    # model = Lasso()
# Setting data variables
    X = housing_train.values  # Shape: (1460, 79)
    y = pd.read_csv("train_prices.csv").values  # Shape: (1460, 1)
    test_x = housing_test.values  # Shape: (1459, 79)
    price_true = pd.read_csv("train_prices.csv")
# train_test_split validation
    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 79)    # y_train.shape = (978, 1)
    grid_search(X_train, y_train)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_val)
    # y_val = y_val.reshape(y_val.shape[:1])
    # print(metric(y_val, y_pred))
    #
    # model.fit(X, y)
    # price_pred = model.predict(test_x)
    # index = housing_test.index
    # pred_df = pd.DataFrame(index=index)
    # # pred_df['SalePrice'] = price_pred
    # # pred_df.to_csv("Submission.csv")


# EXTRAS #


# Returns all negatives predicted (and their indices):
    # for i in range(len(pred_df.SalePrice)):
    #     if pred_df.get_value(i, 0, takeable=True) < 0:
    #         print(i, ", ", pred_df.get_value(i, 0, takeable=True))


# TESTING #

# Model: RandomForest
    # Metric: MedianAbs: 11,549.0 - 11,189.2 - 11,876.35
    # Metric: MeanAbs: 19,100.3680498
    # Metric: MeanSq: 982,556,345.949 - 1,358,389,203.78
    # Metric: ExplVar: 0.841691548145
    # Metric: MeanSqLog: 0.021788982229 - 0.0235023178184 <---- WINNER
    # Metric: R2: 0.838998381662

# Model: Lasso
    # Metric: MedianAbs: 16,029.5013744
    # Metric: MeanAbs: 21,906.9826906
    # Metric: MeanSq: 1,192,649,897.97
    # Metric: ExplVar: 0.838225609045
    # Metric: MeanSqLog: ERROR
    # Metric: R2: 0.837544432739

# Model: LogReg
    # Metric: MedianAbs: 26,250.0
    # Metric: MeanAbs: 40,846.033195
    # Metric: MeanSq: 4,173,891,592.33
    # Metric: ExplVar: 0.43569662705
    # Metric: MeanSqLog: 0.0959748166812
    # Metric: R2: 0.431457691422

# Model: LinReg
    # Metric: MedianAbs: 16,023.6214121
    # Metric: MeanAbs: 21,910.0993854
    # Metric: MeanSq: 1,193,264,065.64
    # Metric: ExplVar: 0.83814249713
    # Metric: MeanSqLog: ERROR
    # Metric: R2: 0.837460774527
