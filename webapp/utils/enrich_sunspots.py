import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from webapp.utils.trends_util import rolling_mean, find_minimums


def fill_values(data, ndx, func):
    """ fill values from array :attr:`data` using function :attr:`func` """
    vals = [np.ones((ndx[j + 1] - ndx[j])) * func(data[ndx[j]:ndx[j + 1]])
            for j in range(len(ndx) - 1)]
    result = hstack(vals)
    return result


def get_enriched_dataframe(csf_file="data/solarn_month.csv"):
    """
       enrich dataframe with 1y, 3y and 128 months moving averages and
       with min, max and average number of sunspots
    """
    df = pd.read_csv(csf_file, delimiter=";")
    trend = df['sunspots'].values
    # calculate moving average
    df["mean_1y"] = rolling_mean(df['sunspots'], 12)
    df["mean_3y"] = rolling_mean(df['sunspots'], 36)
    df["mean_12y"] = rolling_mean(df['sunspots'], 128)
    # fill the first value of 96.7 instead of NA
    df["mean_1y"] = df["mean_1y"].fillna(96.7)
    df["mean_3y"] = df["mean_3y"].fillna(96.7)
    df["mean_12y"] = df["mean_12y"].fillna(96.7)
    # find minimums in trend using period = 128 months
    mins = find_minimums(trend, 128)
    # correction for short cycle after minimum #7 using period = 119 months
    correction = find_minimums(trend[mins[7]:(mins[7] + 120)], 119)
    # next cycle after minimum #7
    m = mins[7] + correction[1]
    # correction for many zeroes at the end of minimum #5
    k = (mins[5] + mins[6]) // 2
    # drop invalid minimums 6 and 9
    indices = [0] + mins[:5] + [k, mins[7], m, mins[8]] + mins[10:] +\
              [len(trend)]
    # calculate min, max and average number of sunspots for solar cycles
    min_ = fill_values(trend, indices, np.min)
    max_ = fill_values(trend, indices, np.max)
    avg = fill_values(trend, indices, np.mean)
    df["sn_mean"] = pd.Series(avg.tolist())
    df["sn_max"] = pd.Series(max_.tolist())
    df["sn_min"] = pd.Series(min_.tolist())

    y_max = hstack([np.zeros([indices[17]]),
                    np.ones((indices[20] - indices[17])),
                    np.zeros([indices[-1] - indices[20]])])
    y_min = hstack([np.zeros([indices[5]]),
                    np.ones((indices[8] - indices[5])),
                    np.zeros([indices[-1] - indices[8]])])
    df["y_min"] = pd.Series(y_min.tolist())
    df["y_max"] = pd.Series(y_max.tolist())
    return df


def predict_cv_and_plot_results(clf, params, data, df):
    """ predict_cv_and_plot_results """
    y_max = df["y_max"].values
    y_min = df["y_min"].values
    # Initialize a stratified split of our dataset for the validation process
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    gcv = GridSearchCV(clf, params, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data, y_max)
    pred_max = gcv.predict(data) * 60
    gcv = GridSearchCV(clf, params, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data, y_min)
    pred_min = gcv.predict(data) * 100
    class_name = str(clf.__class__)[(str(clf.__class__).rfind(".") + 1):-2]
    plt.figure(figsize=(16, 5))
    plt.title(f"{class_name} prediction")
    plt.plot(df['year_float'].values, df["sn_max"].values)
    plt.plot(df['year_float'].values, df["mean_1y"].values)
    plt.plot(df['year_float'].values, df["mean_12y"].values)
    plt.plot(df['year_float'].values, pred_max)
    plt.plot(df['year_float'].values, pred_min)
    plt.show()


def find_best_classifier():
    """ find best classifier """
    df = get_enriched_dataframe()
    cols = ["sunspots", "observations", "mean_1y", "mean_3y", "mean_12y", "sn_mean", "sn_max", "sn_min"]

    data_scaled = StandardScaler().fit_transform(df[cols].values)
    params_lr = {'C': np.linspace(8, 200, 20) / 10, 'class_weight': ["balanced", None]}
    params_rid = {'alpha': np.linspace(8, 200, 20) / 10, 'class_weight': ["balanced", None]}
    params = {"n_estimators": [3, 4, 5, 7], "max_depth": [3, 4, 5, 6, 9]}
    estimators = {"n_estimators": [4, 5, 7, 10, 12]}
    params_dt = {"max_depth": [3, 4, 5, 6, 9]}
    params_knn = {"n_neighbors": [4, 5, 7, 8, 10, 12]}
    params_ada = {"n_estimators": [3, 4, 5, 7], "learning_rate": [0.2, 1., 9.9]}
    classifiers = [
        (LogisticRegression(), params_lr),
        (RidgeClassifier(), params_rid),
        (GradientBoostingClassifier(), params),
        (BaggingClassifier(DecisionTreeClassifier()), estimators),
        (DecisionTreeClassifier(), params_dt),
        (RandomForestClassifier(), params),
        (KNeighborsClassifier(), params_knn),
        (AdaBoostClassifier(), params_ada),
        (ExtraTreesClassifier(), params),
    ]
    for clf, parameters in classifiers:
        predict_cv_and_plot_results(clf, parameters, data_scaled, df)
