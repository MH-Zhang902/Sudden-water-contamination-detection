# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:05:17 2020

@author: 12406
"""
import module3
import module4 as m4
import module5
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.collections as collections


path = r"C:\Users\12406\Desktop\test\论文代码\Excel\江西南昌滁槎pH残差序列.xlsx"
#载入时间序列数据
data = pd.read_excel(path,index_col=0)
#d = data.iloc[:,0:5]
#print(d)

dates = data.index
distance_threshold = 'None'
s = 3
n_clusters = 5
t = 15
tp = '倒U型'
st = 150
sp = 100
n = int((len(dates)-st)/sp)+1
xlist = []
for i in range(n):
    a = st-1+sp*i
    b = t
    xlist.append((a,b))

labels,_ = m4.Hiera_Cluster(data,n_clusters,distance_threshold)
y_truth,y_pred,list1,list2 = module5.Get_res(data,labels,t)

    
#set value for initial probability of an event
pe0=1e-5
#set smoothing parameter 0.3<alpha<0.9
alpha=0.6
#Iterate Bayse Update Rule
pe=pe0
P_event = []
tp = fn = fp = tn = 0
for i in range(len(y_pred)):
    if y_truth[i]==y_pred[i]==0:
        tp += 1
    elif y_truth[i]==0 and y_pred[i]==1:
        fn += 1
    elif y_truth[i]==y_pred[i]==1:
        tn += 1
    elif y_truth[i]==1 and y_pred[i]==0:
        fp += 1
        #print(tp, fp, fn, tn)
    tpr = 0 if (tp+fn)==0 else tp/(tp+fn) #检出率
    fpr = 0 if (fp+tn)==0 else fp/(fp+tn)  #误报率
    temp_err=y_pred[i]
        #2 Bayse calculations - outlier and non-outlier
    if temp_err == 0:
        pe1=pe
        #outlier Bayse rule - TP and FP from get_TP_FP.m
        pe=tpr*pe/(tpr*pe+fpr*(1-pe))
        #smoothing
        pe=alpha*pe+(1-alpha)*pe1
        #eliminate convergence to 1
        pe=min(pe,0.95)
    else:
        pe1=pe;
        #non-outlier Bayse rule - TP and FP from get_TP_FP.m
        pe=(1-tpr)*pe/((1-tpr)*pe+(1-fpr)*(1-pe))
        #smoothing
        pe=alpha*pe+(1-alpha)*pe1
        #eliminate convergence to 0
        pe=max(pe,pe0)
    #save probability of event
    P_event.append(pe)

df = pd.DataFrame({'事件发生概率':P_event},index=dates)
df.to_excel(path[:-5] + '事件发生概率.xlsx',encoding='utf-8')
#print(x)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#colors = ['aqua', 'darkorange', 'cornflowerblue','navy','deeppink','red','black','green','darkgray']
fig, ax = plt.subplots(figsize=(8,3),dpi=200)
ax.plot(dates, P_event, color='red')
ax.set_ylim([-0.05,1.05])
ax.set_ylabel('P_event',fontsize=12)
ax.set_xlabel('时间',fontsize=12)
ax.axhline(0.7, linestyle='--',color='black')
ax.set_xticks(['2019-10-30-00h00m','2019-11-30-00h00m','2019-12-30-00h00m','2020-01-30-00h00m','2020-02-30-00h00m'])
ax.tick_params(labelsize=8)
#ax.legend(loc='right')
collection = collections.BrokenBarHCollection(xlist, (0,1),facecolor='green', alpha=0.5)
ax.add_collection(collection)

plt.show()
