# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:05:41 2020

@author: Administrator
"""


import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.metrics import RocCurveDisplay,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression



#读入数据
def Read_dataset(path):
    df = pd.read_excel(path)A
    return df

#获取污染事件检测结果
def Get_res(datas,labels,t):
    y_truth = []
    start =150
    l = len(datas)
    for i in range(l):
        if 0 <= (i-start)%100 < t and i>=start-1:
            y_truth.append(0)
        else:
            y_truth.append(1)
    tuple1 = []
    for y in range((l-start)//100):
        x1 = start + 100*y
        x2 = start + t -1 + 100*y
        tuple1.append((x1,x2))#模拟的污染事件
    #print(y_truth)
    dict1 = {}
    y_pred = []
    for each in labels:
        dict1[each] = dict1.get(each,0) + 1
    keys = sorted(dict1, key=lambda k: dict1[k])
    for j in range(len(labels)):
        if labels[j] == keys[-1]:
            y_pred.append(1)
        else:
            y_pred.append(0)
    tuple2 = []
    tuple3 = []
    tuple2 = [k for k,x in enumerate(y_pred) if x == 0]
    for a in range(len(tuple2)):
        if a == 0:
            tuple3.append(tuple2[a])
        elif tuple2[a] - tuple2[a-1] > 1:
            tuple3.append(tuple2[a-1])
            tuple3.append(tuple2[a])
    try:
        tuple3.append(tuple2[-1])
    except:
        print('检测无异常')
    #print(tuple3)
    #print(list1)
    list1 = []
    for c in range(len(tuple3)):
        if c%2 == 0:
            list1.append((tuple3[c],tuple3[c+1]))
    return y_truth,y_pred,tuple1,list1

    #去除孤立事件
def Remove_Isolatedevent(list1):
    list2 = []
    for i in range(len(list1)):
            if list1[i][1] - list1[i][0] >= 2:
                list2.append(list1[i])
    return list2

#输出污染事件检测对比图
def Plot_comparation(dates,superimposed_datas,X,Y,T_ratio,F_ratio,aver_lenth,aver_timelength,aver_overlaptime,path):
    line = Line(init_opts=opts.InitOpts(width="1000px", height="400px"))\
    .add_xaxis(dates)\
    .add_yaxis(
        series_name="叠加污染事件数据",
        y_axis=superimposed_datas,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(color='green'))\
    .set_global_opts(
        title_opts=opts.TitleOpts(title=path+"\n聚类法检测结果"),
        graphic_opts=[
                opts.GraphicGroup(
                    graphic_item=opts.GraphicItem(
                        # 控制整体的位置
                        left="70%",
                        top="10%",
                    ),
                    children=[
                          # opts.GraphicRect控制方框的显示
                        # 如果不需要方框，去掉该段即可
                        opts.GraphicRect(
                            graphic_item=opts.GraphicItem(
                                z=100,
                                left="center",
                                top="middle",
                            ),
                            graphic_shape_opts=opts.GraphicShapeOpts(
                                width=190, height=90,
                            ),
                            graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                fill="#fff",
                                stroke="#555",
                                line_width=2,
                                shadow_blur=8,
                                shadow_offset_x=3,
                                shadow_offset_y=3,
                                shadow_color="rgba(0,0,0,0.3)",
                            )
                        ),
                        # opts.GraphicText控制文字的显示
                        opts.GraphicText(
                            graphic_item=opts.GraphicItem(
                                left="center",
                                top="middle",
                                z=100,
                            ),
                            graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                                # 可以通过jsCode添加js代码，也可以直接用字符串
                                text='T_ratio={}\nF_ratio={}\naver_length={}\naver_timelength={}\naver_overlaptime={}'
                                .format(T_ratio,F_ratio,aver_lenth,aver_timelength,aver_overlaptime),
                                font="14px Microsoft YaHei",
                                graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                    fill="#333"
                                )
                            )
                        )
                    ])],
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_left="right"),
        xaxis_opts=opts.AxisOpts(boundary_gap=False),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)))
    list1 = []
    for i in X:
        list1.append(opts.MarkAreaItem(x=i,y=(0,6),itemstyle_opts=opts.ItemStyleOpts(color='grey',opacity=0.2)))
    for j in Y:
        list1.append(opts.MarkAreaItem(x=j,y=(0,4),itemstyle_opts=opts.ItemStyleOpts(color='red')))
    line.set_series_opts(markarea_opts=opts.MarkAreaOpts(data=list1))
    line.render(path+"聚类法检测结果.html")

#输出ROC曲线
def Plot_ROC(superimposed_datas,y_truth,y_pred):
    cm = confusion_matrix(y_truth, y_pred)
    #cm_display = ConfusionMatrixDisplay(cm,display_labels=['0','1'])
    #cm_display.plot(values_format='d') #画出混淆矩阵图
    tp, fn, fp, tn = cm.ravel()
    #print(tn, fn, fp, tp)
    TPR = tp/(tp+fn)  #检出率
    FPR = fp/(fp+tn)  #误报率
    #print(TPR,FPR)
    #logistic_model = LogisticRegression()
    #clf = logistic_model.fit(np.array(superimposed_datas).reshape(len(superimposed_datas),1),y_truth)
    #probability = clf.decision_function(np.array(superimposed_datas).reshape(len(superimposed_datas),1))
    #print(probability)
    #fpr, tpr, thresholds = roc_curve(y_pred,probability)
    #print(fpr, tpr, thresholds)
    #roc_auc= auc(fpr, tpr)
    #display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='ROC')
    #display.plot()#画出ROC曲线图
    return '{:.2f}'.format(TPR), '{:.2f}'.format(FPR)#,'{:.2f}'.format(roc_auc)

#污染事件检出率
def Event_Detected_Res(list1,list2,t):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for i,j in list1:
        list4 = []
        for a,b in list2:
            if b-a >= 0 and (i <= a < j or i < b <= j):
                list4 = [i,j,a,b]
                list4.sort()
                count1 += 1
                count2 += b-a
                count3 += a-i
                count4 += list4[2]-list4[1]
    count5 = len(list2)-count1
    True_ratio = min(count1/len(list1),1)#事件检出率
    False_ratio = count5/len(list1)
    if True_ratio == 0:
        aver_length = 0
        aver_timelength = t
        aver_overlaptime = 0
    else:
        aver_length = count2/count1
        aver_timelength = count3/count1
        aver_overlaptime = count4/count1
    #print(ratio)
    return '{:.2f}'.format(True_ratio),'{:.2f}'.format(False_ratio),'{:.2f}'.format(aver_length),'{:.2f}'.format(aver_timelength),'{:.2f}'.format(aver_overlaptime)

if __name__ == '__main__':
    path = './TOC江西南昌滁槎(s=3&t=15&tp=倒U)聚类结果.xlsx'
    df = Read_dataset(path)
    t = int(input('请输入模拟的污染事件时长：'))
    y_truth,y_pred,list1,list2 = Get_res(df['预处理后数据'].tolist(), df['聚类结果'].tolist(),t)
    #print(list1,list2)
    print(list2)
    list3 = Remove_Isolatedevent(list2)
    print(list3)
    ratio,aver_length,aver_timelength,aver_overlaptime = Event_Detected_Res(list1,list2,t)
    TPR, FPR = Plot_ROC(df['叠加污染事件'],y_truth,y_pred)
    Plot_comparation(df.index.tolist(),df['叠加污染事件'].tolist(),list1,list2,ratio,aver_length,aver_timelength,aver_overlaptime)
    print(ratio,aver_length,aver_timelength,aver_overlaptime)