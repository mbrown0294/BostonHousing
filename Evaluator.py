import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer , mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
from sklearn.model_selection import train_test_split, GridSearchCV


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
    for comp in range(1, 196):
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
    # Let's get started
    housing_train = pd.read_csv("featurized_train.csv")
    housing_test = pd.read_csv("featurized_test.csv")
    housing_test.set_index('Id', drop=True, inplace=True)  # Maintains 'Id' values
    for column in housing_train.columns:
        if column not in housing_test.columns:
            housing_train.drop(column, 1, inplace=True)
    # print(housing_test.shape)  # Test Shape: (1459, 195)
    # print(housing_train.shape)  # Train Shape: (1460, 195)
    metric = mean_squared_log_error
    model = RandomForestRegressor()

    # Setting data variables
    X = housing_train.values  # Shape: (1460, 79)
    y = pd.read_csv("train_prices.csv").values  # Shape: (1460, 1)
    test_x = housing_test.values  # Shape: (1459, 79)
    # y_col = pd.read_csv("train_prices.csv")

    # train_test_split validation
    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 79)    # y_train.shape = (978, 1)

    # grid_search(X_train, y_train)

    # # Create and Print MatPlot of component cost scores at each n_components value
    # component_list, score_list = comp_score_plot(X_train, y_train, X_val, y_val)
    # plt.plot(component_list, score_list)
    # plt.title('Score per # of Principal Components')
    # plt.ylabel('Valid Scores')
    # plt.xlabel('Number of Principal Components')
    # plt.show()

    # Starts the evaluation
    pca = PCA(n_components=31, whiten=True)
    X_train_pca = pca.fit_transform(X_train)    # Shape: (978, n_components)
    y_train = np.squeeze(y_train)  # Shape: (978,)
    # Fit the model and transform
    model.fit(X_train_pca, y_train)
    new_x_val = pca.transform(X_val)  # Shape: (482, n_components)
    y_val_pred = model.predict(new_x_val)  # Shape: (482,)
    y_val = y_val.reshape(y_val.shape[:1])  # Shape: (482,)
    print(metric(y_val, y_val_pred))

    X_pca = pca.fit_transform(X)
    y = np.squeeze(y)
    model.fit(X_pca, y)
    new_test_x = pca.transform(test_x)

    price_pred = model.predict(new_test_x)
    index = housing_test.index
    pred_df = pd.DataFrame(index=index)
    pred_df['SalePrice'] = price_pred
    pred_df.to_csv("Submission.csv")

# Other models
    # model = Lasso()

# Returns all negatives predicted (and their indices):
    # for i in range(len(pred_df.SalePrice)):
    #     if pred_df.get_value(i, 0, takeable=True) < 0:
    #         print(i, ", ", pred_df.get_value(i, 0, takeable=True))
