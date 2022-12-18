# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sys
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier as xgb

from xgboost import XGBClassifier

# 读取带标签的数据
# total_pd = pd.read_csv("../data/embedding_file/deepwalk_wiki_embeding_label.csv")
total_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label_extinfo.csv")

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1.0)
# 重置ID号
total_pd=total_pd.reset_index()
total_pd=total_pd.drop("node_id",axis=1)
# 训练数据Y
train_target_y = total_pd['label']
# 训练数据X
train_data_x = total_pd.drop('label',axis=1)
# 标准化处理
train_data_x_scale = preprocessing.scale(train_data_x)

X_train, X_test, y_train, y_test = train_test_split(train_data_x_scale, train_target_y, test_size=0.2, random_state=42)

#For classification #Random Search
# xgb_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier',XGBClassifier())])
# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }
# random_search = RandomizedSearchCV(xgb_pipeline, param_distributions=params, n_iter=100,
#                                    scoring='f1_weighted', n_jobs=4, verbose=3, random_state=1001 )
# random_search.fit(X_train,y_train)

#OR#Grid Search
model = XGBClassifier()
# 默认参数
gbm_param_grid = {
    'learning_rate': np.array([0.1]),
    'n_estimators': np.array([100]),
    'subsample': np.array([1]),
    'max_depth': np.array([3])
    #'classifier__colsample_bytree': np.arange(0,1.1,.2)
}

# 第二轮参数调优
gbm_param_grid2 = {
    'learning_rate': np.array([0.001, 0.005]),
    'n_estimators': np.array([400, 500, 600]),
    'subsample': np.array([0.9, 0.95, 1.0]),
    'max_depth': np.array([3,5,7,9,10]),
    #'classifier__colsample_bytree': np.arange(0,1.1,.2)
}

# 第一轮的参数调优
gbm_param_grid1 = {
    'learning_rate': np.array([0.01,0.001]),
    'n_estimators': np.array([100,200,300,400]),
    'subsample': np.array([0.7,0.8,0.9]),
    'max_depth': np.array([10,11,12,13,14,15,16,17]),
    'reg_lambda': np.array([1]),
    'gamma': np.array([0])
    #'classifier__colsample_bytree': np.arange(0,1.1,.2)
}

grid_search = GridSearchCV(estimator=model, param_grid=gbm_param_grid, n_jobs= -1,
                         scoring='f1_weighted', verbose=10)

grid_search.fit(X_train,y_train)

#Print out best parameters
#print(random_search.best_params_)
print(grid_search.best_params_)
#Print out scores on validation set
#print(random_search.score(X_test,y_test))
print(grid_search.score(X_test,y_test))