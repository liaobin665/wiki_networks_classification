# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao
# 计算 偏度(skewness)与峰度(kurtosis)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ext_pd = pd.read_csv("../data/embedding_file/Struc2Vec_wiki_embeding_label_extinfo.csv")

degree = ext_pd['pagerank']
print(degree.skew())
print(degree.kurt())