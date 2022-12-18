# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# Extract feature importances
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 这是没有加额外节点结构信息的数据来源
# total_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label.csv")

total_pd = pd.read_csv("../data/embedding_file/sdne_wiki_embeding_label_extinfo.csv",
                      usecols=['node_id','label','degree','in_degree','out_degree','clustering','pagerank'])

# 这是加了节点额外结构信息的数据来源
# total_pd = pd.read_csv("../data/embedding_file/wiki_embeding_with_lable_extinfo.csv")

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1.0)
# 重置ID号
total_pd=total_pd.reset_index()
total_pd=total_pd.drop("index",axis=1)

total_pd=total_pd.drop("node_id",axis=1)
train_target_y = total_pd['label']
train_data_x = total_pd.drop('label',axis=1)

from sklearn import preprocessing

train_data_x_scale = preprocessing.scale(train_data_x)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=200,         # 树的个数--1000棵树建立xgboost
    max_depth=6,               # 树的深度
    min_child_weight = 1,      # 叶子节点最小权重
    gamma=0.,                  # 惩罚项中叶子结点个数前的参数
    subsample=0.8,             # 随机选择80%样本建立决策树
    colsample_btree=0.8,       # 随机选择80%特征建立决策树
    objective='multi:softmax', # 指定损失函数
    scale_pos_weight=1,        # 解决样本个数不平衡的问题
    random_state=27            # 随机数
)
xgb_model.fit(train_data_x_scale,train_target_y)

# 特征重要性
feature_importance_values = xgb_model.feature_importances_
print("*************************")
print(feature_importance_values)

print("-----------------------------------")
print(pd.DataFrame(feature_importance_values).skew())
print(pd.DataFrame(feature_importance_values).kurt())
print("-----------------------------------")
sns.distplot(feature_importance_values,rug = True,
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "g"},  # 设置箱子的风格、线宽、透明度、颜色,风格包括：'bar', 'barstacked', 'step', 'stepfilled'
             kde_kws={"color": "r", "linewidth": 1, "label": "feature importance",'linestyle':'--'},   # 设置密度曲线颜色，线宽，标注、线形
             rug_kws = {'color':'r'} )  # 设置数据频率分布颜色
plt.show()