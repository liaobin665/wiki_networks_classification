# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# 训练得到 wiki node2vec embedding文件
import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier
from GraphEmbedding.ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd

G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

# 入度信息
degree = np.array(G.degree())
in_degree = np.array(G.in_degree())
out_degree = np.array(G.out_degree())
clustering = nx.clustering(G)
pagerank = nx.pagerank(G)

node_stat = pd.DataFrame(data={'e_node_id': degree[:,0], 'degree':degree[:,1], 'in_degree':in_degree[:,1],'out_degree':out_degree[:,1],'clustering':list(clustering.values()) ,'pagerank':list(pagerank.values())})

# 修改node_id数据类型
node_stat[['e_node_id']]=node_stat[['e_node_id']].astype(int)

node_stat=node_stat.sort_values('e_node_id',axis=0)


node_stat.to_csv("../data/embedding_file/wiki_graph_node_ext_info.csv", index=None)
