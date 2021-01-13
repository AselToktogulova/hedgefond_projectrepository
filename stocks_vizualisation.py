import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from help_methods import normalize_data
import os

dir_data= './data/selected_stocks2'

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
            df_temp = normalize_data(df_temp)
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
    df = normalize_data(df)
    dr = df.copy()
    dr.cumsum()
    return dr

dates = pd.date_range('2016-01-01','2020-08-01',freq='B')

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

df = stocks_data(dir_data, dates)
df.fillna(method='pad')
df.interpolate().plot()
plt.tight_layout()
plt.show()

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
df = stocks_data(dir_data, dates, 'SP500.csv')
dr = daily_return(df)
dr = dr.interpolate()
wdi_daily_return = dr
df = stocks_data(dir_data, dates, 'AAPL.csv')
dr = daily_return(df)
dr = dr.interpolate()
tesla_daily_return = dr
result = pd.concat([tesla_daily_return, wdi_daily_return], axis=1)
result.plot(color=['orange', 'green'])
plt.yticks(np.arange(-0.5, 2.5, 0.5))
plt.title('Daily Returns', fontsize=20)
plt.tight_layout()
plt.show()

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


dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df = stocks_data(dir_data, dates, 'SP500.csv')
wire_roc = get_ROC(df, 1)
df = stocks_data(dir_data, dates, 'AAPL.csv')
tesla_roc = get_ROC(df, 1)
result = pd.concat([tesla_roc, wire_roc], axis=1)
result.plot(color=['orange', 'green'])
plt.yticks(np.arange(-0.5, 2.5, 0.5))
plt.title('Rate of Change', fontsize=20)
plt.tight_layout()
plt.show()



dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df_wdi = stocks_data(dir_data, dates, 'WDI.DE.csv')
dr_wdi = daily_return(df_wdi)
df_tesla = stocks_data(dir_data, dates, 'TSLA.csv')
dr_tesla = daily_return(df_tesla)
plt.hist([[x[0] for x in dr_tesla.values.tolist()], [x[0] for x in dr_wdi.values.tolist()]], bins=30, color=['blue','red'])
plt.legend()
plt.title('Daily Returns as a histogram', fontsize=20)
plt.tight_layout()
plt.show()


dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df_wdi = stocks_data(dir_data, dates, 'SP500.csv')
dr_wdi = daily_return(df_wdi)
df_tesla = stocks_data(dir_data, dates, 'AAPL.csv')
dr_tesla = daily_return(df_tesla)
plt.hist([[x[0] for x in dr_tesla.values.tolist()], [x[0] for x in dr_wdi.values.tolist()]], bins=30, color=['orange', 'green'])
plt.legend()
plt.title('Daily Returns as a histogram', fontsize=20)
plt.tight_layout()
plt.show()

dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df_tesla = stocks_data(dir_data, dates, 'TSLA.csv')
df_wdi = stocks_data(dir_data, dates, 'WDI.DE.csv')
result = pd.concat([cum_return(df_tesla), cum_return(df_wdi)], axis=1)
result.plot(color=['blue', 'red'])
plt.title('Cumulative Returns', fontsize=20)
plt.tight_layout()
plt.show()

df_tesla = stocks_data(dir_data, dates, 'SP500.csv')
df_wdi = stocks_data(dir_data, dates, 'AAPL.csv')
result = pd.concat([cum_return(df_tesla), cum_return(df_wdi)], axis=1)
result.plot(color=['orange', 'green'])
plt.title('Cumulative Returns', fontsize=20)
plt.tight_layout()
plt.show()

dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df_wdi = stocks_data(dir_data, dates, 'SP500.csv')
dr_wdi = cum_return(df_wdi)
df_tesla = stocks_data(dir_data, dates, 'AAPL.csv')
dr_tesla = cum_return(df_tesla)
plt.hist([[x[0] for x in dr_tesla.values.tolist()], [x[0] for x in dr_wdi.values.tolist()]], bins=30, color=['green', 'orange'])
plt.legend()
plt.title('Cumulative Returns as a histogram', fontsize=20)
plt.tight_layout()
plt.show()

dates = pd.date_range('2020-01-01','2020-08-01',freq='B')
df_wdi = stocks_data(dir_data, dates, 'WDI.DE.csv')
dr_wdi = cum_return(df_wdi)
df_tesla = stocks_data(dir_data, dates, 'TSLA.csv')
dr_tesla = cum_return(df_tesla)
plt.hist([[x[0] for x in dr_tesla.values.tolist()], [x[0] for x in dr_wdi.values.tolist()]], bins=30, color=['blue','red'])
plt.legend()
plt.title('Cumulative Returns as a histogram', fontsize=20)
plt.tight_layout()
plt.show()

dates = pd.date_range('2015-01-05','2020-12-17',freq='B')
df = pd.read_excel('./data/periodenoverview.xlsx')
df = pd.DataFrame(df, columns=['Date', 'Total Return (P)'])
#df['Date'].values

plt.plot(dates, np.cumsum(df['Total Return (P)'].values))
plt.tight_layout()
plt.show()


dates = pd.date_range('2016-01-01','2020-08-01',freq='B')
df_sp = stocks_data(dir_data, dates, 'SP500.csv')

dates = pd.date_range('2016-01-01','2020-08-01',freq='B')
df_sp = stocks_data(dir_data, dates, 'SP500.csv')

'''
### not used anymore 
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