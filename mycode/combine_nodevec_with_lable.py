# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao



# 实现对lable数据，embedding数据，还有分类数据的合并
import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier
from GraphEmbedding.ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
# node2vec 文件
# embeding_pd = pd.read_csv("../data/embedding_file/wiki_embeding_node2vec.csv",index_col='node_id')
# LINE embedding文件
# embeding_pd = pd.read_csv("../data/embedding_file/LINE_wiki_embeding.csv",index_col='node_id')

# Struc2Vec embedding文件
embeding_pd = pd.read_csv("../data/embedding_file/Stuc2Vec_wiki_embeding.csv",index_col='node_id')

embeding_pd=embeding_pd.sort_index(axis=0,ascending=True)

print(embeding_pd.head(5))
#embeding_pd.sort_values(by='node_id',axis=0,ascending=True)
#embeding_pd.reset_index(drop=True)
label_pd=pd.read_csv("../data/crawl_wiki/crawl_wiki_labels.txt", delimiter=" ", names=['node_id','label'])

# category_pd=pd.read_csv("../data/crawl_wiki/crawl_wiki_category.txt", delimiter=" ",names=['cnode_id','category'])

# 拼接数据
full_pd = pd.concat([embeding_pd,label_pd],axis=1)
#删除多余的id列
#full_pd=full_pd.drop('node_id', axis=1)

#full_pd=full_pd.drop('cnode_id', axis=1)

# full_pd.to_csv('../data/embedding_file/wiki_embeding_with_label_LINE.csv',index=0)
full_pd.to_csv('../data/embedding_file/wiki_embeding_with_label_Struc2Vec.csv',index=0)
print("down")


print(full_pd.shape)