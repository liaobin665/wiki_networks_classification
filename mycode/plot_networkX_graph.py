# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier
from GraphEmbedding.ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

G = nx.read_edgelist("../data/crawl_wiki/crawl_wiki_edgelist.txt",create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

# pos = nx.layout.spring_layout(G)
pos = nx.layout.random_layout(G)

node_sizes = [0.8 + 0.2 * i for i in range(len(G))]
M = G.number_of_edges()
nodes_number =G.number_of_nodes()

edge_colors = range(2, M + 2)
node_colors = range(1,nodes_number+20)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    width=2,
)
# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
pc.set_array(edge_colors)
plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()