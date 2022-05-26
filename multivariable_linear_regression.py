# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:43:44 2022

@author: Chunlin Xiong
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("D:\\python学习\\Python数据挖掘与机器学习实战\\Python数据挖掘与机器学习实战\\code\\第3章 回归分析\\3.4\\Advertising.csv",index_col=0)



#利用散点图可视化关系, height 大小， aspect 比例， kind='reg'可以添加一条最佳拟合曲线
#和95%的置信带
def draw_scatter_with_each_variable():
    sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales',
                 height=7,aspect=0.8, kind='reg')
    plt.show()
 
feature_cols = ['TV','radio','newspaper']
#使用列表选择原始DataFrame的子集
X = data[feature_cols]
#X = data[['TV','radio','newspaper']]
y = data['sales']
#y = data.sales


#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#训练模型
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
model = linreg.fit(X_train,y_train)

#zip不能直接输出，转换成list
print(list(zip(feature_cols,linreg.coef_)))

#用模型预测
y_pred = linreg.predict(X_test)

#评价
#平均绝对误差（Mean Absolute Error, MAE）
#均方误差（Mean Squared Error, MSE）
#均方根误差（Root Mean Squared Error, RMSE）

from sklearn import metrics
import numpy as np
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean = 