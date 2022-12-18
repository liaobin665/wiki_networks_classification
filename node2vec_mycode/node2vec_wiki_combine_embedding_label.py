# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao

# 将embedding文件 和 label文件合并
import pandas as pd

embedding_pd = pd.read_csv("../data/embedding_file/node2vec_wiki_embeding.csv",index_col='node_id')
# 对node_id进行排序
embedding_pd = embedding_pd.sort_index(axis=0, ascending=True)
# 读取label数据
label_pd=pd.read_csv("../data/crawl_wiki/crawl_wiki_labels.txt", delimiter=" ", names=['node_id','label'])

# 拼接数据
full_pd = pd.concat([embedding_pd, label_pd], axis=1)

full_pd.to_csv('../data/embedding_file/node2vec_wiki_embeding_label.csv',index=0)

print("down")



