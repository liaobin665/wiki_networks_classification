# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ext_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label_extinfo.csv")
# print(ext_pd.head(10))
# # sns.distplot(ext_pd['degree'])
# # # sns.pairplot(ext_pd,vars=["degree", "in_degree","out_degree","clustering"], kind = 'reg', diag_kind="kde", palette="husl")
# # sns.pairplot(ext_pd,vars=["degree", "in_degree","out_degree"], kind = 'reg', diag_kind="kde",
# #              hue="label",palette="husl",markers=None)


fig = plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)

sns.distplot(ext_pd['clustering'],rug = True,
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "g"},  # 设置箱子的风格、线宽、透明度、颜色,风格包括：'bar', 'barstacked', 'step', 'stepfilled'
             kde_kws={"color": "r", "linewidth": 1, "label": "clustering",'linestyle':'--'},   # 设置密度曲线颜色，线宽，标注、线形
             rug_kws = {'color':'r'} )  # 设置数据频率分布颜色

ax1 = plt.subplot(1,2,2)
sns.distplot(ext_pd['pagerank'],rug = True,
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "g"},  # 设置箱子的风格、线宽、透明度、颜色,风格包括：'bar', 'barstacked', 'step', 'stepfilled'
             kde_kws={"color": "r", "linewidth": 1, "label": "pagerank",'linestyle':'--'},   # 设置密度曲线颜色，线宽，标注、线形
             rug_kws = {'color':'r'} )  # 设置数据频率分布颜色


plt.show()