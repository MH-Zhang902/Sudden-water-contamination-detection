# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:50:17 2021

@author: 12406
"""


import numpy as np
import pandas as pd
from pyecharts import options as opts
import module5 as m5
from sklearn.metrics import confusion_matrix
import matplotlib.collections as collections
import matplotlib.pyplot as plt


path1 = r"C:\Users\12406\Desktop\test\论文代码\Excel\江西南昌滁槎pH残差序列(倒U型).xlsx"
path2 = r"C:\Users\12406\Desktop\test\论文代码\Excel\江西南昌滁槎pH残差序列.xlsx"
#载入时间序列数据
data1 = pd.read_excel(path1,index_col=0)
data2 = pd.read_excel(path2,index_col=0)
X1 = data1['残差序列']
X2 = data2['残差序列']
dates = data1.index

st = 150
sp = 100
st1,et = 150,862#预测开始的时间和结束时间
x = X1[st1:]
t = 15
datas_mean = np.mean(X2)
datas_std = np.std(X2,ddof=1)
#print(datas_mean,datas_std)


y_truth = []#真实的异常结果
l = len(x)
for i in range(l):
    if 0 <= i%100 < t:
        y_truth.append(1)
    else:
        y_truth.append(0)
        
y_pred = []#三倍标准差阈值法的异常识别结果
for i in range(len(x)):
    if x[i] >= (3*datas_std + datas_mean) or x[i] <= (datas_mean - 3*datas_std):
        y_pred.append(1)
    else:
        y_pred.append(0)
#print(y_pred)

cm = confusion_matrix(y_truth, y_pred)
tp, fn, fp, tn = cm.ravel()
TPR = tp/(tp+fn)  #检出率
FPR = fp/(fp+tn)  #误报率
#print(TPR,FPR)

n = int((len(X1)-st)/sp)+1
xlist = []
for i in range(n):
    a = st-1+sp*i
    b = 15
    xlist.append((a,b))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(figsize=(8,3),dpi=200)
#fig.text(0.06,0.5,'pH(mg/L)',va='center', rotation='vertical',fontsize=14)
ax.set_ylabel('pH(mg/L)',fontsize=12)
#fig.text(0.5,0.05,'时间序列', va='center', ha='center',fontsize=12)
ax.set_xlabel('时间序列',fontsize=12)
ax.plot(dates, X1, color='black',label='残差序列\nTPR={:.2f}\nFPR={:.2f}'.format(TPR,FPR),lw=1)
ax.set_ylim([-1,2.5])
ax.axhline(0, linestyle='--',color='black', lw=1)
ax.legend(loc='upper right')
ax.set_xticks(['2019-10-25-00h00m','2019-11-25-00h00m','2019-12-25-00h00m','2020-01-25-00h00m','2020-02-25-00h00m'])
ax.tick_params(labelsize=8)
d = np.arange(0,len(dates),1)
y_pred = np.array([0]*150+y_pred)
#print(y_pred)
l = y_pred[d]
collection = collections.BrokenBarHCollection.span_where(
    d, ymin=0,ymax=0.9,where=l>0,facecolor='green')
ax.add_collection(collection)
collection = collections.BrokenBarHCollection(xlist, (0,1.5),facecolor='grey', alpha=0.5)
ax.add_collection(collection)

plt.show()