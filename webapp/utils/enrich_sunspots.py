"""
    Apache License 2.0 Copyright (c) 2023 Pavel Bystrov
    enrich sunspots utility
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, \
    train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from webapp.utils import dataframe_util


def predict_using_cross_validation(clf, params, data, dframe):
    """ predict with classifier using cross-validation with n=3 """
    if clf is None or data is None or dframe is None:
        raise ValueError("Empty parameters")
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


def evaluate_classifier(clf, params, data_scaled, dframe):
    """ evaluate classifier """
    if clf is None or dframe is None or params is None or dframe is None:
        raise ValueError("Empty parameters")
    clf.set_params(**params)
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


def get_classifiers_and_params() -> list:
    """ get classifiers and params """
    params_lr = {'C': np.linspace(8, 200, 20) / 10,
                 'class_weight': ["balanced", None]}
    params_rid = {'alpha': np.linspace(8, 200, 20) / 10,
                  'class_weight': ["balanced", None]}
    params = {"n_estimators": [3, 4, 5, 7], "max_depth": [3, 4, 5, 6, 9]}
    estimators = {"n_estimators": [4, 5, 7, 10, 12]}
    params_knn = {"n_neighbors": [4, 5, 7, 8, 10, 12]}
    params_ada = {"n_estimators": [3, 4, 5, 7],
                  "learning_rate": [0.2, 1., 9.9]}
    classifiers = [
        (AdaBoostClassifier(), params_ada),
        (LogisticRegression(), params_lr),
        (RidgeClassifier(), params_rid),
        (GradientBoostingClassifier(), params),
        (BaggingClassifier(DecisionTreeClassifier()), estimators),
        (RandomForestClassifier(), params),
        (KNeighborsClassifier(), params_knn),
        (ExtraTreesClassifier(), params)]
    return classifiers


def get_results_for_best_classifier():
    """ get results for best classifier """
    dframe = dataframe_util.get_enriched_dataframe()
    times = dframe["year_float"].values
    cols = ["sunspots", "observations", "mean_1y", "mean_3y", "mean_12y",
            "sn_mean", "sn_max", "sn_min"]
    data_scaled = StandardScaler().fit_transform(dframe[cols].values)
    max_score = 0.
    results = (ExtraTreesClassifier(), {})
    classifiers = get_classifiers_and_params()
    for clf, parameters in classifiers:
        score, best_params = predict_using_cross_validation(clf,
                                                            parameters,
                                                            data_scaled,
                                                            dframe)
        if max_score < score:
            max_score = score
            results = (clf, best_params)

    print(f"best model {str(results[0].__class__)} ")
    predict_max, predict_min, max_, sunspots = evaluate_classifier(results[0],
                                                                   results[1],
                                                                   data_scaled,
                                                                   dframe)
    return times, predict_max, predict_min, max_, sunspots
