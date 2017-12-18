import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer , mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search(trainx, trainy, valx, valy):
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]
    clf = GridSearchCV(estimator=svm.SVR(), param_grid=parameter_candidates, n_jobs=-1)
    trainy = np.squeeze(trainy)
    clf.fit(trainx, trainy)
    b_score = clf.best_score_
    b_c = clf.best_estimator_.C
    b_kernel = clf.best_estimator_.kernel
    b_gamma = clf.best_estimator_.gamma
    print('Best score: ', b_score)
    print('C=', b_c)
    print('kernel="', b_kernel, '"')
    print('gamma=', b_gamma)
    print('Val Score: ', clf.score(valx, valy), "\n")
    return b_c, b_kernel, b_gamma, b_score


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
        plt.plot(components, scores)
        plt.title('Score per # of Principal Components')
        plt.ylabel('Valid Scores')
        plt.xlabel('Number of Principal Components')
        plt.show()


def time_elapsed(start, end):
    elapsed = end - start
    day_tup = divmod(elapsed.total_seconds(), 86400)
    days = day_tup[0]
    hour_tup = divmod(day_tup[1], 3600)
    hours = hour_tup[0]
    min_tup = divmod(hour_tup[1], 60)
    minutes = min_tup[0]
    seconds = min_tup[1]
    if days > 0:
        print("\nDays: ", days)
    if hours > 0:
        print("Hours: ", hours)
    if minutes > 0:
        print("Minutes: ", minutes)
    print("Seconds: ", seconds)


if __name__ == "__main__":
    time_start = datetime.now()
    print(str(time_start), "\n")
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
    X_test = housing_test.values  # Shape: (1459, 79)

    # train_test_split validation
    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 195) / X_val.shape = (482, 195)
    y_train = np.squeeze(y_train)  # Shape: (978,)
    y_val = np.squeeze(y_val)  # Shape: (482,)
    quick_X = X[:300]
    quick_y = y[:300]
    quick_xval = X[-300:]
    quick_yval = y[-300:]

    # # Grid Search
    # Sample Size/Time Run:
    #     50/1 min, 35.570 sec
    #     100/6 min, 32.175 sec # Up about 5
    #     150/12 min, 38.260 sec # By 6? Maybe linear?
    #     300/ Eh, I gave up. Just run the full train?

    # # On to grid searching
    # best_c, best_kernel, best_gamma, best_score = grid_search(quick_X, quick_y, quick_xval, quick_yval)  # Quick(-ish)
    best_c, best_kernel, best_gamma,  best_score = grid_search(X_train, y_train, X_val, y_val)  # Train (slow)

    '''
    Size 50: C=1/kernel='linear'/gamma='auto'/Best Score: 0.696984862476
    Size 100: C=1/kernel='linear'/gamma='auto'/Best Score: 0.645307692251
    Size 150: C=100/kernel='linear'/gamma='auto'/Best Score: 0.762856090685
    '''

    # svr = svm.SVR(C=best_c, kernel=best_kernel, gamma=best_gamma)

    # # svr.fit(quick_X, quick_y)
    # # # print("SVR Score: ", svr.score(quick_xval, quick_yval))
    # svr.fit(X_train, y_train)
    # # print("SVR Score: ", svr.score(X_val, y_val))

    # # Line graph for n_components and scores
    # comp_score_plot(X_train, y_train, X_val, y_val)

# # # # FILLER COMMENT FOR REMOVED CODE # # #

    # # Finishing Code??
    # svr.fit(X, y)
    # prediction = svr.predict(X_test)
    # index = housing_test.index
    # pred_df = pd.DataFrame(index=index)
    # pred_df['SalePrice'] = prediction
    # # pred_df.to_csv("Submission.csv")
    # print(pred_df)

    time_done = datetime.now()
    print("\n", str(time_done), "\n")
    time_elapsed(time_start, time_done)

    '''
    pca = PCA(n_components=31, whiten=True)
    X_train_pca = pca.fit_transform(X_train)    # Shape: (978, n_components)
    model.fit(X_train_pca, y_train)
    new_x_val = pca.transform(X_val)  # Shape: (482, n_components)
    y_val_pred = model.predict(new_x_val)  # Shape: (482,)

    y_val = y_val.reshape(y_val.shape[:1])  # Shape: (482,)
    # print("Pred: \n", y_val_pred, "\nGround Truth: \n", y_val)
    quick_pred_df = pd.DataFrame({'Price': y_val_pred})
    quick_pred_df.to_csv('Quick_Pred.csv')
    quick_truth_df = pd.DataFrame({'Price': y_val})
    quick_truth_df.to_csv('Quick_Truth.csv')
    # print(metric(y_val, y_val_pred))

    X_pca = pca.fit_transform(X)
    y = np.squeeze(y)
    model.fit(X_pca, y)
    new_test_x = pca.transform(test_x)
    '''
