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
total_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label.csv")
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

features = list(train_data_x.columns)

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

def plot_feature_importances(df, most_important_number=15):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:most_important_number]))),
            df['importance_normalized'].head(most_important_number),
            align='center', edgecolor='k',color='deepskyblue')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:most_important_number]))))
    ax.set_yticklabels(df['feature'].head(most_important_number))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Struc2Vec Feature Importances')
    plt.show()
    return df

# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances,20)
