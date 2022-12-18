# -*- coding: utf-8 -*-
# Copyright (C) 2020.8
# @Author  : zhangtao
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score,classification_report
import matplotlib.pyplot as plt

iris = datasets.load_iris()
n_class = len(set(iris.target))  # 类别数量

print(n_class)

x, y = iris.data, iris.target
y_one_hot = label_binarize(y, np.arange(n_class))  # 转化为one-hot

# 建模
model = LogisticRegression()
model.fit(x, y)

# 预测值y的三种形式
y_score = model.predict(x)  # 形式一：原始值（0或1或2）
y_score_pro = model.predict_proba(x)  # 形式二：各类概率值
y_score_one_hot = label_binarize(y_score, np.arange(n_class))  # 形式三：one-hot值

print("stop")

obj1 = confusion_matrix(y, y_score)
print('confusion_matrix\n', obj1)

print('accuracy:{}'.format(accuracy_score(y, y_score)))
print('precision:{}'.format(precision_score(y, y_score,average='micro')))
print('precision - macro:{}'.format(precision_score(y, y_score,average='macro')))
print('recall:{}'.format(recall_score(y, y_score,average='micro')))
print('f1-score:{}'.format(f1_score(y, y_score,average='micro')))


from sklearn.metrics import roc_curve

# AUC值
auc = roc_auc_score(y_one_hot, y_score_pro, average='micro')

# 画ROC曲线
fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score_pro.ravel())  # ravel()表示平铺开来
plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1.1, 0, 1.1])
plt.xlabel('False Postivie Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

ans = classification_report(y,y_score,digits=5)
print(ans)