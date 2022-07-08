#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: models.py 
@create: 2020/10/9 17:01 
"""

from sklearn.externals import joblib
import numpy as np
from by_stage.ml_baselines.evaluation import eval_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def svr_regression(X_train, X_test, y_train, y_test):
    # svr = GridSearchCV(SVR(), param_grid={"kernel": ['rbf', 'sigmoid'],
    #                                       "C": np.logspace(-3, 3, 7),
    #                                       "gamma": np.logspace(-3, 3, 7)})
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    y_pred = np.around(y_pred)
    return eval_regression(y_test, y_pred)


def bayesian_ridge_regression(X_train, X_test, y_train, y_test):
    # br = GridSearchCV(linear_model.BayesianRidge(),
    #                   param_grid={"alpha_1": [i / 10000.0 for i in range(1, 100, 5)],
    #                               "alpha_2": [i / 10000.0 for i in range(1, 100, 5)]})
    br = linear_model.BayesianRidge()
    br.fit(X_train, y_train)
    y_pred = br.predict(X_test)
    y_pred = np.around(y_pred)
    return eval_regression(y_test, y_pred)


def linear_regression(X_train, X_test, y_train, y_test):
    linear_r = linear_model.LinearRegression()
    linear_r.fit(X_train, y_train)
    y_pred = linear_r.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(linear_r, 'saved_model/linear_all_3.pickle')
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
    lasso_r = linear_model.Lasso(alpha=0.1)
    lasso_r.fit(X_train, y_train)
    y_pred = lasso_r.predict(X_test)
    y_pred = np.around(y_pred)
    return eval_regression(y_test, y_pred)


def gbr_regression(X_train, X_test, y_train, y_test, times=1):
    # params = {
    #     'learning_rate': [0.001, 0.0001,],
    #     'n_estimators': range(20, 50, 10),
    #     'max_depth': range(10, 50, 10)
    # }
    # gbr = RandomizedSearchCV(GBR(),
    #                          param_distributions=params,
    #                          cv=5, n_iter=10, n_jobs=-1, random_state=42)
    # gbr = GBR()
    gbr = GBR(n_estimators=40, max_depth=40)
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(gbr, 'saved_model/gbr_all_filtered_' + str(times) + '.pickle')
    return eval_regression(y_test, y_pred)


def light_gbm(X_train, X_test, y_train, y_test, times=1):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 128,
        'learning_rate': 0.1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 20,
        'num_iterations': 80,
    }
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50,
                    valid_sets=lgb_train)
    y_pred = gbm.predict(X_test)
    joblib.dump(gbm, 'saved_model/gbm_all_filtered_' + str(times) + '.pickle')
    return eval_regression(y_test, y_pred)


def light_gbm_cv(X_train, X_test, y_train, y_test, times=3):
    gbm = lgb.LGBMRegressor(boosting_type='rf',
                            objective='regression',
                            n_estimators=100,
                            max_depth=60,    # 100
                            num_leaves=12000,   # 16000
                            subsample=0.95,
                            bagging_freq=10)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    joblib.dump(gbm, 'saved_model/gbm_emb_1_' + str(times) + '.pickle')
    return eval_regression(y_test, y_pred)


def gbr_regression2(X_train, X_test, y_train, y_test, times=1):
    # params = {
    #     'learning_rate': [0.001, 0.0001,],
    #     'n_estimators': range(20, 50, 10),
    #     'max_depth': range(10, 50, 10)
    # }
    # gbr = RandomizedSearchCV(GBR(),
    #                          param_distributions=params,
    #                          cv=5, n_iter=10, n_jobs=-1, random_state=42)
    # gbr = GBR()
    gbr = GBR(n_estimators=40, max_depth=20)
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    y_pred = np.around(y_pred)
    joblib.dump(gbr, 'saved_model/gbr_all_' + str(times) + '.pickle')
    return eval_regression(y_test, y_pred)


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


def mlp_regression(X_train, X_test, y_train, y_test, times):
    mlp_r = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 128, 128,))

    mlp_r.fit(X_train, y_train)
    y_pred = mlp_r.predict(X_test)
    y_pred = np.around(y_pred)
    return eval_regression(y_test, y_pred)
