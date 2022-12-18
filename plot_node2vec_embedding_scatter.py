# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier
from GraphEmbedding.ge import Node2Vec
from GraphEmbedding.ge import DeepWalk
from GraphEmbedding.ge import LINE
from GraphEmbedding.ge import SDNE
from GraphEmbedding.ge import Struc2Vec
from sklearn.linear_model import LogisticRegression
import  pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn import preprocessing


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/crawl_wiki/crawl_wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    print("debug_stop")
    X, Y = read_node_label('../data/crawl_wiki/crawl_wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    # markers = ['o', 'x', '+', '^', 'v','.',"<",">","1","2"]
    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker="x")
    plt.legend()

    plt.show()



G=nx.read_edgelist('../data/crawl_wiki/crawl_wiki_edgelist.txt',
                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])

model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=4, use_rejection_sampling=0)
model.train(window_size = 5, iter = 3)


embeddings = model.get_embeddings()

evaluate_embeddings(embeddings)
plot_embeddings(embeddings)



# model = DeepWalk(G, walk_length=10, num_walks=80, workers=4)
# model.train(window_size=5, iter=3)

# model = LINE(G, embedding_size=128, order='second')
# model.train(batch_size=1024, epochs=50, verbose=2)

# model = SDNE(G, hidden_size=[256, 128],)
# model.train(batch_size=3000, epochs=40, verbose=2)

# model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
# model.train()
# embeddings=model.get_embeddings()
# result = evaluate_embeddings(embeddings)
# usecols=['degree','in_degree','out_degree','clustering','pagerank']
# ext_pd = pd.read_csv("../data/embedding_file/wiki_graph_node_ext_info.csv")
# ext_pd = ext_pd.drop('e_node_id',axis=1)
# print(ext_pd.head(10))
# ext_pd_scale = preprocessing.scale(ext_pd.T.values)



# result = evaluate_embeddings(ext_pd_scale)
# print(result)

# plot_embeddings(ext_pd_scale)
