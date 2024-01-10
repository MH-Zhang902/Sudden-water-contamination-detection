# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:35:20 2021

@author: 12406
"""


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import module4
import matplotlib.collections as collections

path = r"C:\Users\12406\Desktop\test\论文代码\Excel\江西南昌滁槎pH残差序列(倒U型).xlsx"
#载入时间序列数据
data = pd.read_excel(path,index_col=0)
X = data['残差序列'].tolist()
dates = data.index

n_clusters = 2
distance_threshold = 'None'
labels,model = module4.Hiera_Cluster(X,n_clusters,distance_threshold)
#module4.Plot_dendrogram(model, truncate_mode='level', p=5)
print(type(labels))

st=150
sp=100
n = int((len(X)-st)/sp)+1
xlist = []
for i in range(n):
    a = st-1+sp*i
    b = 15
    xlist.append((a,b))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig, ax = plt.subplots(figsize=(8,3),dpi=200)
fig.text(0.06,0.5,'pH(mg/L)',va='center', rotation='vertical',fontsize=14)
fig.text(0.5,0.05,'时间序列', va='center', ha='center',fontsize=12)
ax.plot(dates, X, color='black',label='残差序列',lw=1)
#ax.scatter(dates, labels, s=1,label = '聚类结果')
ax.set_ylim([-1,2.5])
ax.axhline(0, linestyle='--',color='black', lw=1)
ax.legend(loc='upper right')
ax.set_xticks(['2019-10-25-00h00m','2019-11-25-00h00m','2019-12-25-00h00m','2020-01-25-00h00m','2020-02-25-00h00m'])
ax.tick_params(labelsize=8)
d = np.arange(0,len(dates),1)
l = labels[d]
collection = collections.BrokenBarHCollection.span_where(
    d, ymin=0,ymax=0.9,where=l<=0,facecolor='green')
ax.add_collection(collection)
collection = collections.BrokenBarHCollection(xlist, (0,1.5),facecolor='grey',alpha=0.5)
ax.add_collection(collection)

plt.show()