""" enrich sunspots utility """
import pandas as pd
import numpy as np
from numpy import hstack
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, \
    train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from webapp.utils.trends_util import rolling_mean, find_minimums


def fill_values(data, ndx, func):
    """ fill values from array :attr:`data` using function :attr:`func` """
    if len(data) < 2:
        return data
    for index_ in ndx:
        if index_ > len(data):
            return np.array([])
    vals = [np.ones((ndx[j + 1] - ndx[j])) * func(data[ndx[j]:ndx[j + 1]])
            for j in range(len(ndx) - 1)]
    result = hstack(vals)
    return result


def get_enriched_dataframe(csv_file="data/sunspot_numbers.csv"):
    """
       enrich dataframe with 1y, 3y and 128 months moving averages and
       with min, max and average number of sunspots
    """
    data = pd.read_csv(csv_file, delimiter=";")
    trend = data['sunspots'].values
    # calculate moving average
    data["mean_1y"] = rolling_mean(data['sunspots'], 12)
    data["mean_3y"] = rolling_mean(data['sunspots'], 36)
    data["mean_12y"] = rolling_mean(data['sunspots'], 128)
    # fill the first value of 96.7 instead of NA
    data["mean_1y"] = data["mean_1y"].fillna(96.7)
    data["mean_3y"] = data["mean_3y"].fillna(96.7)
    data["mean_12y"] = data["mean_12y"].fillna(96.7)
    # find minimums in trend using period = 128 months
    mins = find_minimums(trend, 128)
    # correction for short cycle after minimum #7 using period = 119 months
    correction = find_minimums(trend[mins[7]:(mins[7] + 120)], 119)
    # next cycle after minimum #7
    cy8 = mins[7] + correction[1]
    # correction for many zeroes at the end of minimum #5
    cy6 = (mins[5] + mins[6]) // 2
    # drop invalid minimums 6 and 9
    indices = [0] + mins[:5] + [cy6, mins[7], cy8, mins[8]] + mins[10:] + \
              [len(trend)]
    # calculate min, max and average number of sunspots for solar cycles
    min_ = fill_values(trend, indices, np.min)
    max_ = fill_values(trend, indices, np.max)
    avg = fill_values(trend, indices, np.mean)
    data["sn_mean"] = pd.Series(avg.tolist())
    data["sn_max"] = pd.Series(max_.tolist())
    data["sn_min"] = pd.Series(min_.tolist())

    y_max = hstack([np.zeros([indices[17]]),
                    np.ones((indices[20] - indices[17])),
                    np.zeros([indices[-1] - indices[20]])])
    y_min = hstack([np.zeros([indices[5]]),
                    np.ones((indices[8] - indices[5])),
                    np.zeros([indices[-1] - indices[8]])])
    data["y_min"] = pd.Series(y_min.tolist())
    data["y_max"] = pd.Series(y_max.tolist())
    return data


def predict_using_cross_validation(clf, params, data, dframe):
    """ predict cv and plot results """
    y_max = dframe["y_max"].values
    y_min = dframe["y_min"].values
    x_train1, x_test1, max_train, max_test = \
        train_test_split(data, y_max, test_size=0.1, random_state=9)
    x_train2, x_test2, min_train, min_test = \
        train_test_split(data, y_min, test_size=0.1, random_state=9)
    # Initialize a stratified split of dataset for the validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    gcv1 = GridSearchCV(clf, params, n_jobs=-1, cv=skf, verbose=1)
    gcv1.fit(x_train1, max_train)
    pred_max = gcv1.predict(x_test1)
    mse1 = mean_squared_error(max_test, pred_max)
    gcv = GridSearchCV(clf, params, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(x_train2, min_train)
    pred_min = gcv.predict(x_test2)
    mse2 = mean_squared_error(min_test, pred_min)
    class_name = str(clf.__class__)[(str(clf.__class__).rfind(".") + 1):-2]
    print(f"MSE for maximum using \
         {class_name} = {mse1} , MSE for minimum = {mse2}")
    score = (gcv1.best_score_ + gcv.best_score_) * 0.5
    return score, gcv1.best_params_


def evaluate_classifier(clf, data_scaled, dframe):
    """ evaluate classifier """
    y_max = dframe["y_max"].values
    y_min = dframe["y_min"].values
    max_ = dframe["sn_max"].values
    sunspots = dframe["sunspots"].values
    clf2 = clf
    clf2.fit(data_scaled, y_max)
    predict_max = clf2.predict(data_scaled)
    clf.fit(data_scaled, y_min)
    predict_min = clf.predict(data_scaled)
    return predict_max, predict_min, max_, sunspots


def get_results_for_best_classifier():
    """ get results for best classifier """
    dframe = get_enriched_dataframe()
    times = dframe["year_float"].values
    cols = ["sunspots", "observations", "mean_1y", "mean_3y", "mean_12y",
            "sn_mean", "sn_max", "sn_min"]
    data_scaled = StandardScaler().fit_transform(dframe[cols].values)
    params_lr = {'C': np.linspace(8, 200, 20) / 10,
                 'class_weight': ["balanced", None]}
    params_rid = {'alpha': np.linspace(8, 200, 20) / 10,
                  'class_weight': ["balanced", None]}
    params = {"n_estimators": [3, 4, 5, 7], "max_depth": [3, 4, 5, 6, 9]}
    estimators = {"n_estimators": [4, 5, 7, 10, 12]}
    params_dt = {"max_depth": [3, 4, 5, 6, 9]}
    params_knn = {"n_neighbors": [4, 5, 7, 8, 10, 12]}
    params_ada = {"n_estimators": [3, 4, 5, 7],
                  "learning_rate": [0.2, 1., 9.9]}
    classifiers = [
        (AdaBoostClassifier(), params_ada),
        (LogisticRegression(), params_lr),
        (RidgeClassifier(), params_rid),
        (GradientBoostingClassifier(), params),
        (BaggingClassifier(DecisionTreeClassifier()), estimators),
        (DecisionTreeClassifier(), params_dt),
        (RandomForestClassifier(), params),
        (KNeighborsClassifier(), params_knn),
        (ExtraTreesClassifier(), params),
    ]
    max_score = 0.
    results = (ExtraTreesClassifier(), {})
    for clf, parameters in classifiers:
        score, best_params = predict_using_cross_validation(clf,
                                                            parameters,
                                                            data_scaled,
                                                            dframe)
        if max_score < score:
            max_score = score
            results = (clf, best_params)
    print(f"best model {str(results[0].__class__)} {results[1]}")
    predict_max, predict_min, max_, sunspots = evaluate_classifier(results[0],
                                                                   data_scaled,
                                                                   dframe)
    return times, predict_max, predict_min, max_, sunspots
