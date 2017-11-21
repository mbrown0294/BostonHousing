import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer , mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
from sklearn.model_selection import train_test_split, GridSearchCV


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
    print(clf.best_score_)  # 0.657469669426


def comp_score_plot(trainx, trainy, valx, valy):
    components = []
    scores = []
    for comp in range(1, 80):
        components.append(comp)
        mini_pca = PCA(n_components=comp, whiten=True)
        trainx_pca = mini_pca.fit_transform(trainx)  # Shape: (978, 5)
        trainy = np.squeeze(trainy)
        model.fit(trainx_pca, trainy)
        new_valx = mini_pca.transform(valx)  # Shape: (482, 5)
        valy_pred = model.predict(new_valx)
        valy = valy.reshape(valy.shape[:1])
        scores.append(metric(valy, valy_pred))
    return components, scores


if __name__ == "__main__":
    housing_train = pd.read_csv("featurizedTrain.csv")
    housing_test = pd.read_csv("featurizedTest.csv")
    housing_test.set_index('Id', drop=True, inplace=True)  # Maintains 'Id' values
# Set metric
    # metric = mean_squared_error
    # metric = median_absolute_error
    metric = mean_squared_log_error
    # metric = r2_score
    # metric = explained_variance_score
# Set model
    model = RandomForestRegressor()
    # model = Lasso()
# Setting data variables
    X = housing_train.values  # Shape: (1460, 79)
    y = pd.read_csv("train_prices.csv").values  # Shape: (1460, 1)
    test_x = housing_test.values  # Shape: (1459, 79)
    y_col = pd.read_csv("train_prices.csv")
# train_test_split validation
    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 79)    # y_train.shape = (978, 1)
#     grid_search(X_train, y_train)
    component_list, score_list = comp_score_plot(X_train, y_train, X_val, y_val)
    plt.plot(component_list, score_list)
    plt.title('Score per # of Principal Components')
    plt.xlabel('Valid Scores')
    plt.ylabel('Number of Principal Components')
    plt.show()
    pca = PCA(n_components=40, whiten=True)
    X_train_pca = pca.fit_transform(X_train)    # Shape: (978, 5)
    y_train = np.squeeze(y_train)
    model.fit(X_train_pca, y_train)
    new_x_val = pca.transform(X_val)  # Shape: (482, 5)
    y_val_pred = model.predict(new_x_val)
    y_val = y_val.reshape(y_val.shape[:1])
    metric(y_val, y_val_pred)
    new_test_x = pca.transform(test_x)
    # price_pred = model.predict(new_test_x)
    # index = housing_test.index
    # pred_df = pd.DataFrame(index=index)
    # pred_df['SalePrice'] = price_pred
    # pred_df.to_csv("Submission.csv")


# EXTRAS #


# Returns all negatives predicted (and their indices):
    # for i in range(len(pred_df.SalePrice)):
    #     if pred_df.get_value(i, 0, takeable=True) < 0:
    #         print(i, ", ", pred_df.get_value(i, 0, takeable=True))


'''
TESTING

Model: RandomForest
    Metric: MedianAbs(0): 13,895
    Metric: MeanSq(0): 1,757,965,444.23
    Metric: ExplVar(1): 0.841691548145
    Metric: MeanSqLog(0): 0.0380640543864
    Metric: R2(1): 0.838998381662

Model: Lasso
    Metric: MedianAbs: 15,336.756
    Metric: MeanSq: 1,403,132,097.4
    Metric: ExplVar: 0.838225609045
    Metric: MeanSqLog: 0.0346569581418
    Metric: R2: 0.837544432739  (Approach 1)
    
RandomFor/MeanSqLog/PCA:
    components = 5: 0.0456637532179
    components = 10: 0.0420794780193
    components = 15: 0.0396171025027
    components = 20: 0.0381703190471
    components = 25: 0.0389053574836
    components = 30: 0.0358936883713
    components = 35: 0.0357876535382
    components = 40: 0.0357587856916
    components = 50: 0.0365345471203
    components = 79: 0.0373731851169
        Whiten = True:
        5: 0.0445929599005
        10: 0.0436346260886
        15: 0.0377110578718
        20: 0.0397212720809
        25: 0.0387014142115
        30: 0.0375210273287
        35: 0.0371114498236
        40: 0.0342316306501 <--- WINNER
        45: 0.0362241648409
        79: 0.0385043050604
'''
