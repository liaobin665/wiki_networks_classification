# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao




# 训练得到Struc2Vec_wiki_embeding模型的 embedding文件
import networkx as nx
import pandas as pd
from GraphEmbedding.ge import Struc2Vec

G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
model.train()

embeddings = model.get_embeddings()

embeddings_pd = pd.DataFrame(embeddings).T

embeddings_pd.to_csv("../data/embedding_file/Struc2Vec_wiki_embeding.csv", index_label='node_id')

print("Struc2Vec_wiki_embeding down!")