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
from GraphEmbedding.ge import LINE
from GraphEmbedding.ge import Struc2Vec
G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

# model = LINE(G, embedding_size=128, order='second')
# model.train(batch_size=1024, epochs=50, verbose=2)

model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
model.train()

#model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=4, use_rejection_sampling=0)
#model.train(window_size = 5, iter = 3)

embeddings=model.get_embeddings()
# 需要对embeddings结果进行转置操作
embeddings_pd = pd.DataFrame(embeddings).T
print(embeddings_pd.shape)

embeddings_pd.to_csv("../data/embedding_file/Stuc2Vec_wiki_embeding.csv")

print("down!")

