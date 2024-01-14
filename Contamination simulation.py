# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:57:29 2021

@author: 12406
"""


import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from pyecharts.charts import Line
from pyecharts import options as opts


#创建污染事件
def Creat_event(event_strength,event_steps,event_type):
    i = [1,-1]
    event_data = []
    event_strength = int(event_strength)*datas_std
    S = math.sqrt(2*math.pi)*int(event_strength)
    event_steps = int(event_steps)
    if event_type == tp[0]:
        x = np.linspace(-2, 2, (event_steps-2))
        event_data1 = np.around(S*stats.norm.pdf(x,loc=0,scale=1),decimals=2).tolist()
        event_data =[ i[0]*a for a in event_data1]
        event_data.insert(0,0)
        event_data.append(0)
    elif event_type == tp[1]:
        x = np.linspace(-2, 2, (event_steps-2))
        event_data1 =np.around(S*stats.norm.pdf(x,loc=0,scale=1),decimals=2).tolist()
        event_data =[ i[1]*a for a in event_data1]
        event_data.insert(0,0)
        event_data.append(0)
    elif event_type == tp[2]:
        x = np.linspace(-1, 1, int(event_steps/10)*2)
        event_data1 = np.around(S*stats.norm.pdf(x,loc=0,scale=1),decimals=2).tolist()
        event_data =[ i[-1]*a for a in event_data1]
        for j in range(event_steps-2-int(len(event_data1)/2)):
            if j >= int(len(event_data1)/2):
                event_data.insert(j,i[1]*event_strength)
        event_data.insert(0,0)
        event_data.append(0)
    elif event_type == tp[3]:
        x = np.linspace(-1, 1, int(event_steps/10)*2)
        event_data1 = np.around(S*stats.norm.pdf(x,loc=0,scale=1),decimals=2).tolist()
        event_data =[ i[0]*a for a in event_data1]
        for j in range(event_steps-2-int(len(event_data1)/2)):
            if j >= int(len(event_data1)/2):
                event_data.insert(j,i[0]*event_strength)
        event_data.insert(0,0)
        event_data.append(0)
    elif event_type == tp[4]:
        for i in range(event_steps):
            event_data.append(event_strength)
    elif event_type == tp[5]:
        for i in range(event_steps):
            event_data.append(-1*event_strength)
    return event_data



if __name__ == '__main__':
    tp = ['正高斯','倒高斯','正U型','倒U型','正方型','倒方型']#污染事件类型
    event_strength = 3
    datas_std = 1
    event_steps = 35
    x = np.linspace(1,35,35)   # 定义域
    x_smooth = np.linspace(x.min(), x.max(), 300)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(3,2,sharex=True,figsize=(8,10),dpi=600)
    for i in range(len(tp)):
        a = int(i/2)
        b = int(i%2)
        event_data = Creat_event(event_strength,event_steps,tp[i])
        y_smooth = make_interp_spline(x, event_data)(x_smooth)
        ax[a][b].plot(x_smooth,y_smooth,'g',linewidth=2,label='{}'.format(tp[i]))
        ax[2][0].plot([1,1],[0,3],'g',linewidth=2)
        ax[2][0].plot([35,35],[0,3],'g',linewidth=2)
        ax[2][1].plot([1,1],[0,-3],'g',linewidth=2)
        ax[2][1].plot([35,35],[0,-3],'g',linewidth=2)
        ax[a][b].axhline(y=0,color='grey',linestyle='--')
        ax[a][b].legend(loc="upper right",fontsize=10,frameon=False)
        ax[0][0].set_ylim([-0.5,3.8])
        ax[0][1].set_ylim([-3.5,0.8])
        ax[1][1].set_ylim([-0.5,3.8])
        ax[1][0].set_ylim([-3.5,0.8])
        ax[2][0].set_ylim([-0.5,3.8])
        ax[2][1].set_ylim([-3.5,0.8])
        ax[a][b].set_xlim([0,36])
        ax[a][0].set_ylabel('污染强度',fontsize=10)
        ax[2][b].set_xlabel('时间序列',fontsize=10)
       
        
    plt.show()  # 显示