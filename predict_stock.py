# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:52:20 2022

@author: Chunlin Xiong
"""

import quandl
from sklearn import preprocessing

quandl.ApiConfig.api_key = "DxKdLCtx_yNCH2nX1zUo"

df = quandl.get('WIKI/AAPL', start_date="2020-05-01", end_date="2022-05-01", rows=5)
print(df.head())

quandl.bulkdownload("WIKI/AAPL")