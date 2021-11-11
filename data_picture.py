import numpy as np
import pandas as pd
import sqlalchemy as sa
import os
import configparser
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# generate data pictures for all t
def generate_all_pictures(data):
    # calculate 10-days return
    data['return'] = data.groupby('stockid').apply(lambda x: (x.open.shift(-11) / x.open.shift(-1) - 1).fillna(0)) \
        .reset_index(level=0, drop=True)
    # standardize
    data['return'] = data.groupby('date').apply(
        lambda x: ((x['return'] - x['return'].mean()) / x['return'].std()).fillna(0)) \
        .reset_index(level=0, drop=True)

    factor_list = ['open', 'high', 'low', 'close', 'vwap', 'vol', 'pct', 'turnover', 'freeturn']
    data = data.set_index(['stockid', 'date'])
    date_arr = data.index.get_level_values(1)
    date_list = date_arr.unique()
    for i, t in enumerate(tqdm(date_list)):
        if i < 29:
            continue
        t_1 = date_list[i-29]
        tdata = data[(date_arr >= t_1) & (date_arr <= t)]
        stock_list = tdata.index.get_level_values(0).unique()
        X = []
        Y = []
        asset = []
        for stock in stock_list:
            stock_t = tdata.loc[stock,:]
            if stock_t.shape[0] == 30 and stock_t.isnew.max() == 0 and \
                    stock_t.isst.max() == 0 and stock_t.istrade.max() == 1:
                X.append(stock_t[factor_list].T.values)
                Y.append(stock_t.loc[t, 'return'])
                asset.append(stock)
        X = np.array(X)
        Y = np.array(Y)
        asset = np.array(asset)
        np.save(r'pictures/X_%s.npy' % t.strftime('%Y%m%d'), X)
        np.save(r'pictures/Y_%s.npy' % t.strftime('%Y%m%d'), Y)
        np.save(r'pictures/stock_%s.npy' % t.strftime('%Y%m%d'), asset)




# generate training samples given the date
# all stocks, one picture every %gap% days for the last %n*gap% trading days
def generate_training_picture(data, t, gap, num):
    factor_list = ['open', 'high', 'low', 'close', 'vwap', 'vol', 'pct', 'turnover', 'freeturn']
    stock_list = data.stockid.unique()
    X = []
    Y = []
    data = data.set_index(['stockid', 'date'])
    for stockid in tqdm(stock_list):
        stockdata = data.loc[(stockid, slice(None))]
        stock_picture = stockdata[stockdata.amount > 0][factor_list].T

        datearr = stock_picture.columns
        n = len(datearr[datearr <= t])
        if n <= 30:
            continue
        narr = np.arange(n - 10, max(30, n - 10 - gap*num), -gap)
        for i in narr:
            X.append(stock_picture.iloc[:, i - 30:i].values)
            Y.append(stockdata.loc[datearr[i], 'return'])
    return np.array(X), np.array(Y)

# generate predicting samples given the date array
# all stocks, one picture every day for days between begt and endt
def generate_predict_picture(data,begt,endt):
    factor_list = ['open','high','low','close','vwap','vol','pct','turnover','freeturn']
    stock_list = data.stockid.unique()
    data = data.set_index(['stockid','date'])
    X = []
    Y = []
    for stockid in tqdm(stock_list):
        stockdata = data.loc[(stockid,slice(None))]
        stock_picture = stockdata[stockdata.amount>0][factor_list].T
        datearr = stock_picture.columns
        tarr = datearr[(datearr>=begt)&(datearr<=endt)]
        for t in tarr:
            if len(datearr[datearr<=t])<=30:
                continue
            n = np.argmax(datearr[datearr<=t])
            X.append(stock_picture.iloc[:, n - 30:n].values)
            Y.append([stockid,t])
    return np.array(X), np.array(Y)
