# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# 将embedding，label，extra（出度，入度，pagerank等额外信息） 进行合并
import pandas as pd

embeding_label_pd=pd.read_csv("../data/embedding_file/deepwalk_wiki_embeding_label.csv")
ext_info_pd = pd.read_csv("../data/embedding_file/wiki_graph_node_ext_info.csv")

full_pd = pd.concat([embeding_label_pd, ext_info_pd], axis=1)

full_pd = full_pd.drop('e_node_id', axis=1)

full_pd.to_csv('../data/embedding_file/deepwalk_wiki_embeding_label_extinfo.csv', index=0)

print("down!")