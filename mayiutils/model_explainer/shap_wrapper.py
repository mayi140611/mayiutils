#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: shap_wrapper.py
@time: 2019-05-16 11:03

https://www.zhihu.com/question/23180647
https://blog.dominodatalab.com/shap-lime-python-libraries-part-1-great-explainers-pros-cons/

沙普利值(Shapley value),通过考虑各个代理(agent)做出的贡献,来公平地分配合作收益 代理i的沙普利值是i对于一个合作项目所期望的贡献量的平均值
那么根据沙普利值法,合理分配,是否公平主要考察边际贡献 marginal contribution
"""
import pandas as pd  #for manipulating data
import sklearn  #for building models
from sklearn.model_selection import train_test_split  #for creating a hold-out sample
import sklearn.ensemble  #for building models
import numpy as np  #for manipulating data
import shap  #SHAP package
import xgboost as xgb  #for building models
import matplotlib.pyplot as plt  #visualizing output


if __name__ == '__main__':
    mode = 1
    X, y = shap.datasets.boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # put into a df for exploration
    y_df = pd.DataFrame(y)
    y_df.columns = ["MEDV"]
    housing = X.join(y_df)

    if mode == 1:
        # Build the model
        # xgb_model = xgb.train({'objective': 'reg:linear'}, xgb.DMatrix(X_train, label=y_train))
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(X_train, y_train)

        # Create the SHAP Explainers
        # SHAP has the following explainers: deep, gradient, kernel, linear, tree, sampling

        # Tree on XGBoost
        explainerXGB = shap.TreeExplainer(xgb_model)
        shap_values_XGB_test = explainerXGB.shap_values(X_test)
        shap_values_XGB_train = explainerXGB.shap_values(X_train)

        # Get the SHAP values into dataframes so we can use them later on
        df_shap_XGB_test = pd.DataFrame(shap_values_XGB_test, columns=X_test.columns.values)
        df_shap_XGB_train = pd.DataFrame(shap_values_XGB_train, columns=X_train.columns.values)
        print(df_shap_XGB_test.iloc[0])
        # print(df_shap_XGB_test.head(2))
        # print(df_shap_XGB_train.head(2))

        # Pick an instance to explain
        # optional, set j manually
        # j = 0
        j = np.random.randint(0, X_test.shape[0])
        # initialize js
        # shap.initjs()
        # plt.figure()
        # shap.force_plot(explainerXGB.expected_value, shap_values_XGB_test[j], X_test.iloc[[j]])
        # plt.show()
    # sk_xgb = sklearn.ensemble.GradientBoostingRegressor()
    # sk_xgb.fit(X_train, y_train)
    # rf = sklearn.ensemble.RandomForestRegressor()
    # rf.fit(X_train, y_train)
    # knn = sklearn.neighbors.KNeighborsRegressor()
    # knn.fit(X_train, y_train)




    # Tree on Scikit GBT
    # explainerSKGBT = shap.TreeExplainer(sk_xgb)
    # shap_values_SKGBT_test = explainerSKGBT.shap_values(X_test)
    # shap_values_SKGBT_train = explainerSKGBT.shap_values(X_train)

    # Tree on Random Forest
    # explainerRF = shap.TreeExplainer(rf)
    # shap_values_RF_test = explainerRF.shap_values(X_test)
    # shap_values_RF_train = explainerRF.shap_values(X_train)

    # Must use Kernel method on KNN
    #
    # Summarizing the data with k-Means is a way to speed up the processing
