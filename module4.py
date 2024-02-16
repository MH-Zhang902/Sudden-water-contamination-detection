# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:14:37 2020

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from pyecharts.charts import Line,Scatter
from pyecharts import options as opts

#读入数据
def Read_dataset(path):
    df = pd.read_excel(path)
    return df

#进行层次聚类,输出聚类结果
def Hiera_Cluster(superimposed_datas,n_clusters,distance_threshold):
    if type(superimposed_datas) is list:
        X = np.array(superimposed_datas).reshape(len(superimposed_datas),1)
    else:
        X = np.array(superimposed_datas)
    #print(X)
    n = None
    distance = 0
    if distance_threshold != 'None':
        distance = int(distance_threshold)
        ac=AgglomerativeClustering(n_clusters=n,affinity='euclidean',linkage='average',distance_threshold=distance)
    else:
        distance = None
        n = int(n_clusters)
        ac=AgglomerativeClustering(n_clusters=n,affinity='euclidean',linkage='average',distance_threshold=distance)
    model=ac.fit(X)
    labels = ac.fit_predict(X)
    return labels,model


#画层次聚类树状图
def Plot_dendrogram(model,**kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
#画出聚类树状图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(6,5),dpi=600)
    plt.title('层次聚类树状图')
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("各节点包含的数据量 (无括号的数值为数据点的序号)")
    plt.show()
    #plt.savefig(path[2:-5]+'聚类树状图.jpg')

#存入数据
def Save_toexcel(datas,superimposed_datas,labels,dates):
    datas = datas.tolist()
    superimposed_datas = superimposed_datas.tolist()
    dates = dates.tolist()
    df = pd.DataFrame({'预处理后数据':datas,
                       '叠加污染事件':superimposed_datas,
                       '聚类结果':labels},index=dates)
    df.to_excel(path[2:-5] + '聚类结果.xlsx',encoding='utf-8')

#聚类结果图
def Plot_res(datas,superimposed_datas,labels,dates,name_item,path):
    line = Line(init_opts=opts.InitOpts(width="1000px", height="400px"))
    line.add_xaxis(dates.tolist())
    line.add_yaxis('预处理后数据',datas.tolist(),\
                   linestyle_opts=opts.LineStyleOpts(color='green'))
    line.add_yaxis('叠加污染事件数据',superimposed_datas,\
                   linestyle_opts=opts.LineStyleOpts(color='red',type_='dashed'))
    line.set_global_opts(
            title_opts=opts.TitleOpts(title=path[2:-5]+'\n聚类结果'),
            tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),\
            legend_opts=opts.LegendOpts(pos_left="right"))
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    scatter = Scatter()
    scatter.add_xaxis(dates.to_list())
    scatter.add_yaxis('聚类结果', labels.tolist(),symbol_size=5)
    scatter.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    lines= line.overlap(scatter)
    lines.render('{}'.format(name_item)+path[5:-5] + '聚类结果.html')

if __name__ == '__main__':
    path = './TOC江西南昌滁槎(s=3&t=15&tp=倒U).xlsx'
    df = Read_dataset(path)
    name_item = 'TOC'
    #affinity = input('请从（ “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”）中选择一种：')
    n_clusters = input('最终聚类的簇数：')
    #linkage = input('请从(“ward”, “complete”, “average”, “single”)中选择一种：')
    distance_threshold = input('请输入距离阈值：')
    labels,model = Hiera_Cluster(df['叠加污染事件'],n_clusters,distance_threshold)
    if distance_threshold != 'None':
        Plot_dendrogram(model, truncate_mode='level', p=5)
    Save_toexcel(df['标准化后数据'],df['叠加污染事件'],labels,df.index)
    #print(labels)
    Plot_res(df['标准化后数据'],df['叠加污染事件'],labels,df.index,name_item,path)
