# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

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
# total_pd = pd.read_csv("../data/embedding_file/wiki_embeding_with_label_LINE.csv")

total_pd = pd.read_csv("../data/embedding_file/wiki_embeding_with_lable_extinfo_LINE.csv")
# 这是加了节点额外结构信息的数据来源
# total_pd = pd.read_csv("../data/embedding_file/wiki_embeding_with_lable_extinfo.csv")


total_pd=total_pd.drop("node_id",axis=1)
train_target_y = total_pd['label']
train_data_x = total_pd.drop('label',axis=1)

from sklearn import preprocessing
train_data_x_scale = preprocessing.scale(train_data_x)
#
# import xgboost as xgb
# xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=500,         # 树的个数--1000棵树建立xgboost
#     max_depth=6,               # 树的深度
#     min_child_weight = 1,      # 叶子节点最小权重
#     gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#     subsample=0.8,             # 随机选择80%样本建立决策树
#     colsample_btree=0.8,       # 随机选择80%特征建立决策树
#     objective='multi:softmax', # 指定损失函数
#     scale_pos_weight=1,        # 解决样本个数不平衡的问题
#     random_state=27            # 随机数
# )
# xgb_model.fit(train_data_x_scale,train_target_y)
#
# # 特征重要性
# feature_importance = xgb_model.feature_importances_
# print(feature_importance)

# x_train, x_test, y_train, y_test = train_test_split(train_data,train_target,test_size= 0.2,random_state=0)
# print(x_train.shape)
# print(x_test.shape)
#
# print(y_train.shape)
# print(y_test.shape)
#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)


#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(MLPClassifier())
# classifiers.append(RadiusNeighborsClassifier())
# classifiers.append(RidgeClassifierCV)



#不同机器学习交叉验证结果汇总 scoring='accuracy'
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, train_data_x_scale, train_target_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))


# 求出模型得分的均值和标准差
# 求出模型得分的均值和标准差
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

print(cv_means)

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna', 'MLPClassifier']})
# print(cvResDf)
# LR以及GradientBoostingCla模型在该问题中表现较好。

# 可视化查看不同算法的表现情况
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})
plt.show()