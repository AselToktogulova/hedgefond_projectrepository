import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

import os

# Dec 31, 2015 - Dec 31, 2020 source: https://finance.yahoo.com/
dir_data= './data/selected_stocks'

def get_SMA(df, ndays):
    dm = df.copy()
    return dm.rolling(ndays).mean()

def stocks_data(dir, dates, spec=None):
    df = pd.DataFrame(index=dates)
    if spec == None:
        for file in os.listdir(dir):
            df_temp = pd.read_csv(dir + "/" + file, index_col='Date',
                    parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Close': file})
            df = df.join(df_temp)
    else:
        df_temp = pd.read_csv(dir + "/" + spec, index_col='Date',
                    parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': spec})
        df = df.join(df_temp)
    return df

def daily_return(df):
    dr = df.copy()
    dr = dr[:-1].values / dr[1:] - 1
    return dr

def get_ROC(df, ndays):
    dn = df.diff(ndays)
    dd = df.shift(ndays)
    dr = dn/dd
    return dr

def cum_return(df):
    dr = df.copy()
    dr.cumsum()
    return dr

dates = pd.date_range('2016-01-01','2020-08-01',freq='B')

'''
df = stocks_data(dir_data, dates)
df.fillna(method='pad')
df.interpolate().plot()
plt.tight_layout()
plt.show()
'''
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df = stocks_data(dir_data, dates, 'WDI.DE.csv')


dr = daily_return(df)
dr = dr.interpolate()
wdi_daily_return = dr


df = stocks_data(dir_data, dates, 'TSLA.csv')
dr = daily_return(df)
dr = dr.interpolate()
tesla_daily_return = dr

result = pd.concat([tesla_daily_return, wdi_daily_return], axis=1)

result.plot(color=['blue', 'red'])
plt.title('Daily Returns', fontsize=20)
plt.tight_layout()
plt.show()

dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df = stocks_data(dir_data, dates, 'IBM.csv')


dr = daily_return(df)
dr = dr.interpolate()
wdi_daily_return = dr


df = stocks_data(dir_data, dates, 'AAPL_max.csv')
dr = daily_return(df)
dr = dr.interpolate()
tesla_daily_return = dr

result = pd.concat([tesla_daily_return, wdi_daily_return], axis=1)

result.plot(color=['orange', 'green'])
plt.yticks(np.arange(-0.5, 2.5, 0.5))
plt.title('Daily Returns', fontsize=20)
plt.tight_layout()
plt.show()


'''
dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df = stocks_data(dir_data, dates, 'WDI.DE.csv')
wire_roc = get_ROC(df, 1)

df = stocks_data(dir_data, dates, 'TSLA.csv')
tesla_roc = get_ROC(df, 1)

result = pd.concat([tesla_roc, wire_roc], axis=1)

result.plot(color=['blue', 'red'])
plt.title('Rate of Change', fontsize=20)
plt.tight_layout()
plt.show()
'''

'''
dr.hist(bins=20, color='red')
plt.title('WDI Daily Returns as a histogram', fontsize=20)
plt.show()
'''

'''
dr = cum_return(df)
dr.plot(color='red')
plt.title('WDI Cumulative Returns', fontsize=20)
plt.show()
dr.hist(color='red')
plt.title('WDI Cumulative Returns as a histogram', fontsize=20)
plt.show()

dm = get_SMA(df, 7)
dm.plot(color='red')
plt.title('WDI Simple Moving Average', fontsize=20)
plt.show()

df = stocks_data(dir_data, dates, 'TSLA.csv')
dr = daily_return(df)
dr = dr.interpolate()
dr.interpolate().plot()
plt.title('TSLA Daily Returns', fontsize=20)
plt.show()

dr.hist(bins=20)
plt.title('TSLA Daily Returns as a histogram', fontsize=20)
plt.show()

dr = get_ROC(df, 1)
dr.plot()
plt.title('TSLA Rate of Change', fontsize=20)
plt.show()

dr = cum_return(df)
dr.plot()
plt.title('TSLA Cumulative Returns', fontsize=20)
plt.show()
dr.hist()
plt.title('TSLA Cumulative Returns as a histogram', fontsize=20)
plt.show()


dm = get_SMA(df, 7)
dm.plot()
plt.title('TSLA Simple Moving Average', fontsize=20)
plt.show()

'''