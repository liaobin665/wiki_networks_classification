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
total_pd = pd.read_csv("../data/embedding_file/sdne_wiki_embeding_label_extinfo.csv", usecols=['node_id','label','degree','in_degree','out_degree','clustering','pagerank'])

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1)
# 重置ID号
total_pd=total_pd.reset_index()

total_pd=total_pd.drop("node_id",axis=1)
# 训练数据Y
train_target_y = total_pd['label']
# 训练数据X



train_data_x = total_pd.drop('label',axis=1)
# 标准化处理
train_data_x_scale = preprocessing.scale(train_data_x)

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=5)

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
classifiers.append(XGBClassifier())

#不同机器学习交叉验证结果汇总 scoring='accuracy'
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, train_data_x_scale, train_target_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))

result_pd = pd.DataFrame(cv_results)

# 求出模型得分的均值和标准差
# 求出模型得分的均值和标准差
cv_means = []
cv_min = []
cv_max = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_min.append(cv_result.min())
    cv_max.append(cv_result.max())
    cv_std.append(cv_result.std())

print("模型均值为：")
print(cv_means)

print("模型标准差为：")
print(cv_std)

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'cv_min':cv_min,
                        'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna', 'MLPClassifier','XGBClassifier']})
cvResDf.to_csv("../data/result/wiki_extinfo_alone_accuracy_result.csv")

# 可视化查看不同算法的表现情况
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})
plt.show()