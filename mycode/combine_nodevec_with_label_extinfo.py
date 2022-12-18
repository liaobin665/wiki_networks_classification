# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao


# 实现对lable数据，embedding数据，还有分类数据的合并

import pandas as pd
# node2vec
# embeding_label_pd=pd.read_csv("../data/embedding_file/wiki_embeding_node2vec_with_lable.csv")
# LINE
# embeding_label_pd=pd.read_csv("../data/embedding_file/wiki_embeding_with_label_LINE.csv")

embeding_label_pd=pd.read_csv("../data/embedding_file/wiki_embeding_with_label_Struc2Vec.csv")
ext_info_pd = pd.read_csv("../data/embedding_file/wiki_embeding_node2vec_graph_node_ext_info.csv")

# 拼接数据

full_pd = pd.concat([embeding_label_pd,ext_info_pd],axis=1)
#删除多余的id列
#full_pd=full_pd.drop('node_id', axis=1)

full_pd=full_pd.drop('e_node_id', axis=1)

print("down")

full_pd.to_csv('../data/embedding_file/wiki_embeding_with_lable_extinfo_Struc2Vec.csv',index=0)

print("down")


print(full_pd.shape)