# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# 只包含extra（出入度，pagerank等） 信息进行训练
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

# 读取带标签的数据
# total_pd = pd.read_csv("../data/embedding_file/sdne_wiki_embeding_label.csv")
total_pd = pd.read_csv("../data/embedding_file/sdne_wiki_embeding_label_extinfo.csv",
                       usecols=['node_id','label','degree','in_degree','out_degree','clustering','pagerank'])

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1)
# 重置ID号
total_pd=total_pd.reset_index()

total_pd=total_pd.drop("node_id",axis=1)
# 训练数据Y
train_target_y = total_pd['label']
# 训练数据X

train_target_y.value_counts().plot(kind='bar')

plt.show()
