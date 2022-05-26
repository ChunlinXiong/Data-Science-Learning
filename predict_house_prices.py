# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:25:52 2022

@author: Chunlin Xiong
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# 从csv读取数据


def get_data_from_csv(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet, single_prices_value in zip(data['Size'], data['Price']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append([float(single_prices_value)])
    return x_parameter, y_parameter

# 进行线性回归，并预测predict_value


def linear_model_main(x_parameters, y_parameters, predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(x_parameters, y_parameters)
    predict_outcome = regr.predict([[predict_value]])
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

# 显示线性拟合模型的结果


def show_linear_line(x_parameter, y_parameter):
    regr = linear_model.LinearRegression()
    regr.fit(x_parameter, y_parameter)
    plt.scatter(x_parameter, y_parameter, color='blue')
    plt.plot(x_parameter, regr.predict(x_parameter), color='red', linewidth=4)
    #plt.xticks(())
    #plt.yticks(())
    plt.show()


x_parameter, y_parameter = get_data_from_csv(
    "D:\python学习\input_data_house_price.csv")
predict_value = 700
result = linear_model_main(x_parameter, y_parameter, predict_value)
print("Intercept value ", result['intercept'])
print("coefficient ", result['coefficient'])
print("Predicted value: ", result['predicted_value'])
show_linear_line(x_parameter, y_parameter)
