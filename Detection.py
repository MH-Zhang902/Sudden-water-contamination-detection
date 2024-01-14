# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:54:53 2021

@author: 12406
"""


import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.cluster.hierarchy import dendrogram
from pyecharts.charts import Line,Scatter
from pyecharts import options as opts
import module5 as m5
import module4 as m4
import module3 as m3
from threeδ_threshold import threeδ_threshold
from sklearn.cluster import KMeans

path = r"D:\test\论文代码\Excel\江西南昌滁槎.xlsx"
#载入时间序列数据
data = pd.read_excel(path,index_col=0)
X = data['NH4']
distance_threshold = 'None'
s = [1,1.5,2,2.5,3,4,5,6,7,8,9,10]
n_clusters = 4
t = 15
tp = '倒U型'
st = 200
sp = 100

st1,et = 50,862#预测开始的时间和结束时间
predictions = list()
test = X[st1:]
for i in X[st1-1:et-1]:
    predictions.append(i)


tpr1 = []
fpr1 = []
tpr2 = []
fpr2 = []
tpr3 = []
fpr3 = []
for i in range(len(s)):
    #print(n_clusters[i])
    event_data = m3.Creat_event(X,s[i],t,tp)
    superimposed_data = m3.Superimposed_data(X,event_data,st,sp)#叠加污染事件后数据

    res = []
    for k in range(len(predictions)):
        residual = superimposed_data[st1+k]-predictions[k]
        res.append(residual)
    
    #层次聚类法
    labels1,_ = m4.Hiera_Cluster(res,n_clusters,distance_threshold)
    y_truth1,y_pred1,list1,list2 = m5.Get_res(res,labels1,t)
    TPR1,FPR1= m5.Plot_ROC(res,y_truth1,y_pred1)
    tpr1.append(float(TPR1))
    fpr1.append(float(FPR1))
    
    #3倍标准差阈值法
    labels2 = threeδ_threshold(res)
    y_truth2,y_pred2,list1,list2 = m5.Get_res(res,labels2,t)
    TPR2,FPR2= m5.Plot_ROC(res,y_truth2,y_pred2)
    tpr2.append(float(TPR2))
    fpr2.append(float(FPR2))
    
    #k-means法
    x = np.array(res).reshape(len(res),1)
    kmeans = KMeans(n_clusters=2,init='k-means++', random_state=0).fit(x)
    labels3 = kmeans.labels_
    y_truth3,y_pred3,list1,list2 = m5.Get_res(res,labels3,t)
    TPR3,FPR3= m5.Plot_ROC(res,y_truth3,y_pred3)
    tpr3.append(float(TPR3))
    fpr3.append(float(FPR3))


#print(tpr1,fpr1,tpr2,fpr2,tpr3,fpr3)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#colors = ['aqua', 'darkorange', 'cornflowerblue','navy','deeppink','red','black','green','darkgray']
fig, ax = plt.subplots(figsize=(6,5),dpi=300)
ax.plot(s, tpr1, color='red',label='HC TPR')
ax.plot(s, tpr2, color='black',label='3δ TPR')
ax.plot(s, fpr1, color='green',label='HC FPR')
ax.plot(s, fpr2, color='blue',label='3δ FPR')
ax.set_ylim([-0.05,1.05])
ax.axhline(0, linestyle='--',color='grey',lw=1)
ax.axhline(1, linestyle='--',color='grey', lw=1)
#ax.set_ylabel('',fontsize=16)
ax.set_xlabel('Event intensity(t=15,tp=RVS U)',fontsize=20)
ax.legend(loc='best',fontsize=18)
ax.tick_params(labelsize=16, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                 ) 
plt.show()