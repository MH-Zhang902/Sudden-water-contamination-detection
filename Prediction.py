# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:30:08 2021

@author: 12406
"""


import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error,r2_score


path = r"D:\test\论文代码\Excel\江西南昌滁槎.xlsx"
#载入时间序列数据
data = pd.read_excel(path,index_col=0)
X = data['TOC']
#print(X)
st,et = 150,862#预测开始的时间和结束时间
prediction1 = []
prediction2 = []
prediction3 = []
test = X[st:]
for i in range(st,et):
    train= X[i-st:i]

#TSI预测模型
    prediction1.append(X[i-1])#TSI预测值
    
#AR预测模型
    model1 = AR(train)
    model_fit1 = model1.fit()
    window1 = model_fit1.k_ar
    #print(window)
    coef1 = model_fit1.params
    #print(coef)
# walk forward over time steps in test
    history1 = train[len(train)-window1:]
    history1 = [history1[i] for i in range(len(history1))]
    length = len(history1)
    lag = [history1[i] for i in range(length-window1,length)]
    yhat1 = coef1[0]
    for d in range(window1):
        yhat1 += coef1[d+1] * lag[window1-d-1]
        #obs = test[t]
    prediction2.append(yhat1)#AR预测值

#END-AR预测模型
    decomposer = EMD(train)              
    imfs = decomposer.decompose()
    #print(len(imfs1))  
    predictions = []
    for j in range(len(imfs)):
        model = AR(imfs[j])
        model_fit = model.fit()
        window = model_fit.k_ar
        #print(window)
        coef = model_fit.params
        #print(coef)
        #walk forward over time steps in test
        history = imfs[j][len(imfs[j])-window:]
        history = [history[a] for a in range(len(history))]
        length = len(history)
        lag = [history[b] for b in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
            #obs = test[t]
        predictions.append(yhat)
    pred = sum(predictions)
    prediction3.append(pred)#EMD-AR预测值
error1 = mean_squared_error(prediction1,test)
error2 = mean_squared_error(prediction2,test)
error3 = mean_squared_error(prediction3,test)
r1 = r2_score(prediction1,test)
r2 = r2_score(prediction2,test)
r3 = r2_score(prediction3,test)
print(error1,error2,error3)
print(r1,r2,r3)
#print('Test MSE: %.3f' % error1)
#print('Test R2: %.3f' % r2)

# plot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig,axes = plt.subplots(3,1,sharex=True,figsize=(8,7),dpi=300)
fig.text(0.05,0.5,'TOC(mg/L)',va='center', rotation='vertical',fontsize=24)
fig.text(0.5,0.05,'Time series', va='center', ha='center',fontsize=24)

axes2 = axes[0]
axes2.plot(data.index[st:et],test,'k',label='True value')
axes2.plot(data.index[st:et],prediction1, '--r',label='TSI:MSE=%.3f' % error1)
axes2.set_ylim([0, 20])
axes2.legend(loc='upper right',fontsize=18)
axes2.tick_params(axis='y',
                 labelsize=14, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                 ) 
axes3 = axes[1]
axes3.plot(data.index[st:et],test,'k',label='True value')
axes3.plot(data.index[st:et],prediction2, '--g',label='AR:MSE=%.3f' % error2)
axes3.set_ylim([0, 20])
axes3.legend(loc='upper right',fontsize=16)
axes3.tick_params(axis='y',
                 labelsize=14, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                  ) 
axes4 = axes[2]
axes4.plot(data.index[st:et],test,'k',label='True value')
axes4.plot(data.index[st:et],prediction3, '--b',label='EMD-AR:MSE=%.3f' % error3)
axes4.set_ylim([0, 20])
axes4.legend(loc='upper right',fontsize=14)
axes4.set_xticks(['2019-11-20-00h00m','2019-12-20-00h00m','2020-01-20-00h00m','2020-02-20-00h00m'])
labels = axes4.set_xticklabels(['2019-10-30','2019-11-30','2019-12-30','2020-01-30','2020-02-30'],fontsize = 14) # 设置刻度标签
axes4.tick_params(axis='y',
                 labelsize=14, # y轴字体大小设置
                 color='black',    # y轴标签颜色设置  
                 labelcolor='black', # y轴字体颜色设置
                  ) 
'''
plt.figure(figsize=(8,3))
plt.plot(data.index[st:et],test,'k',label='真实值')
plt.plot(data.index[st:et],prediction1, '--r',label='TSI:MSE=%.3f' % error1)
plt.plot(data.index[st:et],prediction2, '--g',label='AR:MSE=%.3f' % error2)
plt.plot(data.index[st:et],prediction3, '--b',label='EMD-AR:MSE=%.3f' % error1)
#plt.plot((232,232),(0,14),'--g')
plt.ylim([5, 9])
plt.ylabel('pH',fontsize=16)
plt.xticks(['2019-11-15-00h00m','2019-11-30-00h00m','2019-12-30-00h00m','2020-01-30-00h00m','2020-02-30-00h00m'])
#axes5.set_xticklabels(['2019-10-30-00h00m','2019-11-30-00h00m','2019-12-30-00h00m','2020-01-30-00h00m','2020-02-30-00h00m'],fontsize=10)
plt.xlabel('时间',fontsize=16)
plt.legend(loc='upper right')
plt.tight_layout()
'''
plt.show()