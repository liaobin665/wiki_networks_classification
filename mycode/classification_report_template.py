# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao



# 功能：收集不同 算法的，不同性能指标，如：accuracy	precision	recall	F1-score
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
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score,classification_report
# 读取带标签的数据
# total_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label.csv")
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

X_train, X_test, y_train, y_test = train_test_split(train_data_x_scale,train_target_y,test_size=0.2,random_state=0)



#汇总不同模型算法
classifiers=[]
# classifiers.append(SVC())
# classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
# classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression())
# classifiers.append(LinearDiscriminantAnalysis())
# classifiers.append(MLPClassifier())
classifiers.append(XGBClassifier())

accuracy=[]
precision=[]
recall=[]
F1_score=[]

for model in classifiers:
    model.fit(X_train,y_train)
    y_score = model.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_score))
    precision.append(precision_score(y_test,y_score,average="macro"))
    recall.append((recall_score(y_test,y_score,average="macro")))
    F1_score.append(f1_score(y_test,y_score,average="macro"))
    ans = classification_report(y_test, y_score, digits=5)
    print(ans)

print("-----------------------------------------------------------------------------")
# 汇总数据
cvResDf = pd.DataFrame({'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score':F1_score,
                        'algorithm': ['RandomForestClassifier','GradientBoostingClassifier','XGBClassifier']})
print(cvResDf)
# cvResDf.to_csv("../data/result/deepwalk_xgboost.csv")
