# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets

#导入机器学习算法库
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBClassifier

total_pd = pd.read_csv("../data/embedding_file/node2vec_wiki_embeding_label_extinfo.csv")

total_pd=total_pd.drop("node_id",axis=1)
# 训练数据Y
y = total_pd['label']
# 训练数据X
train_data_x = total_pd.drop('label',axis=1)
# 标准化处理
X = preprocessing.scale(train_data_x)

# iris = datasets.load_iris()
# X = iris.data[:, 0:2]  # we only take the first two features for visualization
# y = iris.target

n_features = X.shape[1]

C = 10

# Create different classifiers.
# classifiers = {
#     'L1 logistic': LogisticRegression(C=C, penalty='l1',
#                                       solver='saga',
#                                       multi_class='multinomial',
#                                       max_iter=10000),
#     'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
#                                                     solver='saga',
#                                                     multi_class='multinomial',
#                                                     max_iter=10000),
#     'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
#                                             solver='saga',
#                                             multi_class='ovr',
#                                             max_iter=10000),
#     'Linear SVC': SVC(kernel='linear', C=C, probability=True,
#                       random_state=0),
#     'GPC': GaussianProcessClassifier(kernel)
# }

classifiers = {
    'SVC': SVC(kernel='linear', C=C, probability=True,
                      random_state=0),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'MLPClassifier': MLPClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'XGBClassifier': XGBClassifier()
}

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, y)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))


