# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# 训练得到wiki sdne模型的 embedding文件
import networkx as nx
import pandas as pd

from GraphEmbedding.ge import SDNE
from GraphEmbedding.ge import LINE

G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

model = LINE(G, embedding_size=128, order='second')
model.train(batch_size=1024, epochs=50, verbose=2)

embeddings = model.get_embeddings()

embeddings_pd = pd.DataFrame(embeddings).T

embeddings_pd.to_csv("../data/embedding_file/LINE_wiki_embeding.csv", index_label='node_id')

print("sdne_down!")