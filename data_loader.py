import numpy as np
import pandas as pd
import sqlalchemy as sa
import os
import configparser
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def read_db_config(ini_file='database.ini', section='WIND'):
    """
    Load database.ini to get configuration to connect Oracle database
    """
    if not os.path.exists(ini_file):
        raise IOError('File not exist[%s]' % config_file)

    config = configparser.ConfigParser()
    config.read(ini_file, encoding='utf-8')
    db_config = {}
    if section in config.sections():
        db_config = dict(config._sections[section])
    else:
        print('Section not exist：' + section)

    return db_config


def read_stock_data():
    """
    Read all stock data from 2010-01-01 to 2020-12-31
    -----------------------
    :return: pd.DataFrame
        All stock trading information for every trade day between
        2010-01-01 to 2020-12-31
    -----------------------
    """
    # get configuration
    config = read_db_config()
    # create connection to database
    eng = sa.create_engine(('{dbtype}://{user}:{password}@{host}:{port}/'
                            '{sid}').format(**config))

    # Load market data from Filesync database
    marketdata = pd.DataFrame()

    # Read in a monthly frequency
    datearr = pd.date_range('2009-12-31', '2020-12-31', freq='M')

    for i in tqdm(range(len(datearr) - 1)):
        # begin date and end date
        lastmonthday = datearr[i].strftime("%Y%m%d")
        thismonthday = datearr[i + 1].strftime("%Y%m%d")

        # query 1: price info, including open, high, low, close, pctchange, volume, amount, ...
        query1 = """
            SELECT 
            S_INFO_WINDCODE,
            TRADE_DT,
            S_DQ_OPEN,
            S_DQ_HIGH,
            S_DQ_LOW,
            S_DQ_CLOSE,
            S_DQ_PCTCHANGE,
            S_DQ_VOLUME,
            S_DQ_AMOUNT,
            S_DQ_AVGPRICE,
            S_DQ_ADJFACTOR
            FROM FILESYNC.AShareEODPrices where
            TRADE_DT > '%s' and TRADE_DT <= '%s'
            """ % (lastmonthday, thismonthday)
        data1 = pd.read_sql(query1, eng)

        # query 2: derivative info, including turnover, free turnover, limit status
        query2 = """
            SELECT 
            S_INFO_WINDCODE,
            TRADE_DT,
            S_DQ_TURN,
            S_DQ_FREETURNOVER,
            UP_DOWN_LIMIT_STATUS
            FROM FILESYNC.AShareEODDerivativeIndicator where
            TRADE_DT > '%s' and TRADE_DT <= '%s'
            """ % (lastmonthday, thismonthday)
        data2 = pd.read_sql(query2, eng)
        # merge together
        data1 = pd.merge(data1, data2, how='left', on=['s_info_windcode', 'trade_dt'])
        marketdata = pd.concat([marketdata, data1])
        # sleep for 5 seconds to release database
        time.sleep(5)

    # get ST，*ST stock
    query_st = """
        SELECT 
        S_INFO_WINDCODE,
        S_TYPE_ST,
        ENTRY_DT,
        REMOVE_DT
        FROM FILESYNC.AShareST
        """
    stdata = pd.read_sql(query_st, eng)
    # set current ST,*ST stock remove date to a default date
    stdata.remove_dt = stdata.remove_dt.fillna('20990101')

    # get stock listdate and delistdate
    query_listdate = """
        SELECT 
        S_INFO_WINDCODE,
        S_INFO_LISTDATE,
        S_INFO_DELISTDATE
        FROM FILESYNC.AShareDescription
        """
    listdata = pd.read_sql(query_listdate, eng)
    # drop stocks that have not listed
    listdata = listdata.dropna(subset=['s_info_listdate'])
    # set current stock delistdate to a default date
    listdata.s_info_delistdate = listdata.s_info_delistdate.fillna('20990101')

    # merge marketdata and listdate
    data = pd.merge(marketdata, listdata, on='s_info_windcode', how='left')

    # further merge with ST data
    data = pd.merge(data, stdata, on='s_info_windcode', how='left')

    # add column isst: whether a stock is in ST or *ST
    data.entry_dt = data.entry_dt.fillna('20990101')
    data.remove_dt = data.remove_dt.fillna('20000101')
    data['isst'] = data.apply(lambda x: 1 if (x.trade_dt >= x.entry_dt) & (x.trade_dt <= x.remove_dt) else 0,
                              axis=1)

    # drop additional columns
    data = data.drop(columns=['s_info_delistdate', 's_type_st', 'entry_dt', 'remove_dt'])

    # rename columns to a simple version
    data.columns = ['stockid', 'date', 'open', 'high', 'low', 'close', 'pct', 'vol', 'amount', 'vwap', 'adjfactor',
                    'turnover', 'freeturn',
                    'limit', 'listdate', 'isst']

    # fill nan columns
    data.limit = data.limit.fillna(0)

    # convert date string to datetime
    data.date = pd.to_datetime(data.date)
    data.listdate = pd.to_datetime(data.listdate)

    # add column istrade : whether a stock is trading on that day
    data['istrade'] = data.amount.apply(lambda x: 1 if x > 0 else 0)

    # add column isnew : whether a stock has been traded for at least 120 days
    data['isnew'] = 0
    data.loc[data.date - data.listdate <= pd.Timedelta('120 days'), 'isnew'] = 1

    # save the data as a parquet file
    data.to_parquet('data.parquet')

    return data


def read_index_data():
    """
    Read index data from 2010-01-01 to 2020-12-31
    -----------------------
    :return: pd.DataFrame
        Shanghai Securities Composite Index (000001.SH) trading information for every trade day between
        2010-01-01 to 2020-12-31
    -----------------------
    """
    # get market index data
    query_index = """
        SELECT 
        TRADE_DT,
        S_DQ_CLOSE
        FROM FILESYNC.AIndexEODPrices
        WHERE S_INFO_WINDCODE = '000001.SH'
        AND TRADE_DT>='20100101' AND TRADE_DT<='20201231'
        """
    indexdata = pd.read_sql(query_index, eng)

    # rename columns
    indexdata.columns = ['date', 'close']

    # convert date string to datetime
    indexdata.date = pd.to_datetime(indexdata.date)

    # save the data as a parquet file
    indexdata.to_parquet('marketindex.parquet')

    return indexdata