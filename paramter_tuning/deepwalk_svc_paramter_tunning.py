# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 读取带标签的数据
# total_pd = pd.read_csv("../data/embedding_file/deepwalk_wiki_embeding_label.csv")
total_pd = pd.read_csv("../data/embedding_file/deepwalk_wiki_embeding_label_extinfo.csv")

total_pd=total_pd.drop("node_id",axis=1)
# 训练数据Y
train_target_y = total_pd['label']
# 训练数据X
train_data_x = total_pd.drop('label',axis=1)
# 标准化处理
train_data_x_scale = preprocessing.scale(train_data_x)

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

svcmodel = SVC()
svc_para_grid = [
    {'C':[1,2,5,10,15,20,50,100,1000],
     'kernel':['linear']},
    {'C':[1,2,5,10,15,20,50,100,1000],
     'gamma':[0.001,0.0001],'kernel':['rbf']}
]

modelgsSVC = GridSearchCV(svcmodel,param_grid = svc_para_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)

modelgsSVC.fit(train_data_x_scale,train_target_y)

print('modelgsSVC模型得分为：', modelgsSVC.cv_results_)
print('modelgsSVC模型最佳参数为：', modelgsSVC.best_params_)
print('modelgsSVC模型得分最佳为：', modelgsSVC.best_score_)