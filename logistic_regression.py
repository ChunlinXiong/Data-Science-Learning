# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:27:12 2022

@author: Chunlin Xiong
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math.exp as exp
from pandas import DataFrame

file_name = "D:\\python学习\\Python数据挖掘与机器学习实战\\Python数据挖掘与机器学习实战\\code\\第3章 回归分析\\3.7\\data.txt"

def loadDataSet():
    df=pd.read_csv(file_name)

def sigmoid(inX:int):
    return 1.0/(1.0+exp(-inX))
    
loadDataSet()