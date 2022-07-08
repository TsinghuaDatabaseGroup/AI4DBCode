#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: models.py 
@create: 2020/10/9 17:01 
"""

import heapq
import numpy as np
from by_workload.ml_baselines.evaluation import eval_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


def svr_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, 'saved_model/svr_scaler.pickle')
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train / 100
    # svr = GridSearchCV(SVR(), param_grid={"kernel": ['rbf', 'sigmoid'],
    #                                       "C": np.logspace(-3, 3, 7),
    #                                       "gamma": np.logspace(-3, 3, 7)})
    svr = SVR(kernel='rbf', C=1, gamma='auto')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    y_pred = y_pred * 100
    y_pred = np.around(y_pred)
    joblib.dump(svr, 'saved_model/svr_no_code_1.pickle')
    return eval_regression(y_test, y_pred)


def bayesian_ridge_regression(X_train, X_test, y_train, y_test):
    # br = GridSearchCV(linear_model.BayesianRidge(),
    #                   param_grid={"alpha_1": [i / 10000.0 for i in range(1, 100, 5)],
    #                               "alpha_2": [i / 10000.0 for i in range(1, 100, 5)]})
    br = linear_model.BayesianRidge()
    br.fit(X_train, y_train)
    y_pred = br.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(br, 'saved_model/br_no_code_1.pickle')
    return eval_regression(y_test, y_pred)


def linear_regression(X_train, X_test, y_train, y_test):
    linear_r = linear_model.LinearRegression()
    linear_r.fit(X_train, y_train)
    y_pred = linear_r.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(linear_r, 'saved_model/linear_all_8.pickle')
    return eval_regression(y_test, y_pred)


def sgd_regression(X_train, X_test, y_train, y_test):
    params = {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['elasticnet', 'l1', 'l2'],
        'alpha': np.logspace(-8, 2, 10)
    }
    sgd_r = RandomizedSearchCV(linear_model.SGDRegressor(),
                               param_distributions=params,
                               cv=5, n_iter=20, n_jobs=-1)
    # sgd_r = linear_model.SGDRegressor()
    sgd_r.fit(X_train, y_train)
    y_pred = sgd_r.predict(X_test)
    y_pred = np.around(y_pred)
    return eval_regression(y_test, y_pred)


def lasso_regression(X_train, X_test, y_train, y_test):
    # params = {
    #     "alpha": np.logspace(-3, 3, 7),
    #     "max_iter": [50, 100, 500, 1000],
    #     "tol": np.logspace(-6, 0, 7)
    # }
    # lasso_r = GridSearchCV(linear_model.Lasso(), param_grid=params)
    lasso_r = linear_model.Lasso()
    lasso_r.fit(X_train, y_train)
    y_pred = lasso_r.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(lasso_r, 'saved_model/lasso_no_code_3.pickle')
    return eval_regression(y_test, y_pred)


def gbr_regression(X_train, X_test, y_train, y_test):
    # params = {
    #     'learning_rate': [0.1, 0.001, 0.0001, 0.00001],
    #     'n_estimators': range(10, 200, 20),
    #     'max_depth': range(3, 30, 3)
    # }
    # gbr = RandomizedSearchCV(GBR(),
    #                          param_distributions=params,
    #                          cv=5, n_iter=20, n_jobs=-1, random_state=42)
    gbr = GBR(n_estimators=40, max_depth=20)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    joblib.dump(gbr, 'saved_model/gbr_no_code_1.pickle')
    compare = np.stack((y_test, y_pred), axis=1)
    observe(y_test, y_pred)
    return eval_regression(y_test, y_pred)


def light_gbm(X_train, X_test, y_train, y_test, times=1):
    gbm = lgb.LGBMRegressor(boosting_type='rf',
                            objective='regression',
                            n_estimators=80,
                            max_depth=100,
                            num_leaves=16000,
                            subsample=0.9,
                            bagging_freq=20)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    joblib.dump(gbm, 'saved_model/gbm_13_' + str(times) + '.pickle')
    observe(y_test, y_pred)
    return eval_regression(y_test, y_pred)


def observe(y_test, y_pred):
    y_pred = np.around(y_pred).tolist()
    y_test = y_test.tolist()
    best_target_idx = set(map(y_test.index, heapq.nsmallest(5, y_test)))
    # best_pred_idx = list(map(y_pred.index, heapq.nsmallest(10, y_pred)))
    best_pred_idx = heapq.nsmallest(5, range(len(y_pred)), y_pred.__getitem__)
    best_target = heapq.nsmallest(5, y_test)
    # best_pred = heapq.nsmallest(10, y_pred)
    best_pred = [y_test[i] for i in best_pred_idx]
    print()


def rf_regression(X_train, X_test, y_train, y_test):
    params = {
        'n_estimators': range(5, 50, 10),
        'max_depth': range(3, 10),
        'max_features': range(5, 25, 5),
        'bootstrap': [True, False]
    }
    rf = RandomizedSearchCV(RandomForestRegressor(),
                            param_distributions=params, cv=5, n_iter=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    # rf = RandomForestRegressor(**rand_rf.best_params_)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return eval_regression(y_test, y_pred)


def mlp_regression(X_train, X_test, y_train, y_test):
    mlp_r = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32,), activation='relu', solver='adam', batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.01, max_iter=200, shuffle=True,
        early_stopping=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    mlp_r.fit(X_train, y_train)
    y_pred = mlp_r.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(mlp_r, 'saved_model/mlp_no_code_8.pickle')
    return eval_regression(y_test, y_pred)
