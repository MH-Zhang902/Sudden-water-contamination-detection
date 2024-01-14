# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:01:54 2021

@author: 12406
"""


from scipy.stats import binom


#print("p(x<=3) = {}".format(binom.cdf(k=3,p=0.1,n=10)))


import numpy as np
import pandas as pd
import module5 as m5
import module4 as m4
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import matplotlib.collections as collections

path = r"D:\test\论文代码\Excel\江西南昌滁槎TOC残差序列(倒U型).xlsx"
#载入时间序列数据
data = pd.read_excel(path,index_col=0)
X = data['残差序列'].tolist()
dates = data['残差序列'].index
n_clusters = 5
t = 15
distance_threshold = 'None'
labels,_ = m4.Hiera_Cluster(X,n_clusters,distance_threshold)


st=150
sp=100
n = int((len(X)-st)/sp)+1
xlist = []
for i in range(n):
    a = st-1+sp*i
    b = t
    xlist.append((a,b))
print(xlist)
x = labels.tolist()
start = 10
t=15
r = 1
#pd,far,mttd = [],[],[]
y_pred = []
for k in range(len(x)):
    
    if k<start:
        y_pred.append(0)
    else:
        n = x[k-start:k].count(1)
        if n>=r:
            y_pred.append(1)
        else:
            y_pred.append(0)
#print(y_pred)
_,_,list1,list2 = m5.Get_res(X,y_pred,t)
list3 = []
for i in range(len(list2)):
    a = list2[i][0]
    b = list2[i][1]-list2[i][0]
    list3.append((a,b))
    
PD,FAR,_,MTTD,_ = m5.Event_Detected_Res(list1,list2,t)
print(PD,FAR,MTTD)

'''
pd.append(PD)
far.append(FAR)
mttd.append(MTTD)
print(pd,far,mttd)
#roc_auc= auc(far, pd)
pd = list(map(float,pd))
far = list(map(float,far))
mttd = list(map(float,mttd))
'''
path1 = r"D:\test\论文代码\Excel\江西南昌滁槎TOC残差序列(倒U型)事件发生概率.xlsx"
#载入时间序列数据
data1 = pd.read_excel(path1,index_col=0)
X1 = data1['事件发生概率'].tolist()
y_pred1 = []
l = len(X1)
for j in range(l):
    if X1[j]>=0.5:
        y_pred1.append(0)
    else:
        y_pred1.append(1)
_,_,list4,list5 = m5.Get_res(X1,y_pred1,t)
print(list5)
list6 = []
for i in range(len(list5)):
    a = list5[i][0]
    b = list5[i][1]-list5[i][0]
    list6.append((a,b))
print(list6)
PD1,FAR1,_,MTTD1,_ = m5.Event_Detected_Res(list4,list5,t)
print(PD1,FAR1,MTTD1)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#colors = ['aqua', 'darkorange', 'cornflowerblue','navy','deeppink','red','black','green','darkgray']
fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6),dpi=300)
ax[0].plot(dates,X, color='black',label='Sequence of residuce')
ax[0].set_ylim([-15,10])
ax[0].set_ylabel('TOC(mg/L)',fontsize=24)
#ax.set_xlabel('时间',fontsize=16)
ax[0].axhline(0, linestyle='--',color='black')

#ax.legend(loc='right')
collection1 = collections.BrokenBarHCollection(xlist, (0,2),facecolor='green')
collection2 = collections.BrokenBarHCollection(list3, (0,1),facecolor='red')
ax[0].add_collection(collection1)
ax[0].add_collection(collection2)
ax[0].set_title('BED',x=0.4,y=0.8,fontsize=16)
ax[0].legend(loc='upper right',fontsize=16)
ax[0].tick_params(labelsize=16, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                 ) 
ax[1].plot(dates,X, color='black',label='Sequence of residuce')
#ax.set_ylim([-0.05,1.05])
ax[1].set_ylabel('TOC(mg/L)',fontsize=24)
ax[1].set_xlabel('TIME SERIES',fontsize=24)
ax[1].set_ylim([-15,10])
ax[1].axhline(0, linestyle='--',color='black')
collection3 = collections.BrokenBarHCollection(xlist, (0,2),facecolor='green')
ax[1].add_collection(collection3)
collection4 = collections.BrokenBarHCollection(list6, (0,1),facecolor='red')
ax[1].add_collection(collection4)
ax[1].set_title('Bayesian',x=0.4,y=0.8,fontsize=16)
ax[1].set_xticks(['2019-10-30-00h00m','2019-11-30-00h00m','2019-12-30-00h00m','2020-01-30-00h00m','2020-02-30-00h00m'])
labels1 = ax[1].set_xticklabels(['2019-10-30','2019-11-30','2019-12-30','2020-01-30','2020-02-30'],fontsize = 14) # 设置刻度标签
#ax[1].tick_params(labelsize=8)
ax[1].legend(loc='upper right',fontsize=16)
ax[1].tick_params(labelsize=16, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                 ) 
plt.tight_layout()
plt.show()