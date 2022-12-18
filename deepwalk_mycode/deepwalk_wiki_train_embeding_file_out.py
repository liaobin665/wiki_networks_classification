# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao



# 训练得到wiki deepwalk模型的 embedding文件
import networkx as nx
import pandas as pd

# from GraphEmbedding.ge import SDNE
from GraphEmbedding.ge import DeepWalk

G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

#model = SDNE(G, hidden_size=[256, 128],)
#model.train(batch_size=3000, epochs=40, verbose=2)
model = DeepWalk(G, walk_length=10, num_walks=80, workers=4)
model.train(window_size=5, iter=3)

embeddings = model.get_embeddings()

embeddings_pd = pd.DataFrame(embeddings).T

embeddings_pd.to_csv("../data/embedding_file/deepwalk_wiki_embeding.csv", index_label='node_id')

print("sdne_down!")