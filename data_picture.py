import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_training_picture(data: pd.DataFrame, t: str, gap: int, num: int):
    """
    Generate training samples given the date
    All stocks, one picture every %gap% days for the last %num% trading days
    ---------------------
    :param data: pd.DataFrame, raw trading data
    :param t: str, date string like '2020-12-31'
    :param gap: int, data picture interval
    :param num: total number of days selected
    :return: np.array, 3d array containing all data pictures, should have
        a dimension of (#sample)*9*30
    ---------------------
    """
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
        narr = np.arange(n - 12, max(30, n - 12 - gap*num), -gap)
        for i in narr:
            X.append(stock_picture.iloc[:, i - 30:i].values)
            Y.append(stockdata.loc[datearr[i], 'return'])
    return np.array(X), np.array(Y)


def generate_all_data_pictures(data: pd.DataFrame):
    """Generate data pictures as features and returns as labels for CNN model inputs
    ----------------------------
    :param data: pd.DataFrame
        All stock trading information in China A-Share Market
        From 2010.01.01 to 2020.12.31

    :return:
        No return
        Stock pictures are stored under /pictures
    ----------------------------
    """
    # calculate 10-days return, start from tomorrow (t+1 to t+11)
    data['return'] = data.groupby('stockid').apply(lambda x: (x.open.shift(-11) / x.open.shift(-1) - 1).fillna(0)) \
        .reset_index(level=0, drop=True)

    # standardize
    data['return'] = data.groupby('date').apply(
        lambda x: ((x['return'] - x['return'].mean()) / x['return'].std()).fillna(0)) \
        .reset_index(level=0, drop=True)

    # nine rows in data picture
    factor_list = ['open', 'high', 'low', 'close', 'vwap', 'vol', 'pct', 'turnover', 'freeturn']

    # get all trade days
    data = data.set_index(['stockid', 'date'])
    date_arr = data.index.get_level_values(1)
    date_list = date_arr.unique()

    for i, t in enumerate(tqdm(date_list)):
        # skip first 29 days since data picture needs 30-day history
        if i < 29:
            continue

        # select recent 30 days
        t_1 = date_list[i-29]
        tdata = data[(date_arr >= t_1) & (date_arr <= t)]

        # select stocks
        stock_list = tdata.index.get_level_values(0).unique()

        # X: all stock data pictures in a given day
        X = []
        # Y: all stock returns in a given day (returns for next 10 days), same order as X
        Y = []
        # asset: all stock ticker in a given day, same order as X
        asset = []

        for stock in stock_list:
            stock_t = tdata.loc[stock, :]
            if stock_t.shape[0] == 30 and stock_t.isnew.max() == 0 and \
                    stock_t.isst.max() == 0 and stock_t.istrade.max() == 1:
                X.append(stock_t[factor_list].T.values)
                Y.append(stock_t.loc[t, 'return'])
                asset.append(stock)

        # convert to np.array and stored as .npy files
        np.save(r'pictures/X_%s.npy' % t.strftime('%Y%m%d'), np.array(X))
        np.save(r'pictures/Y_%s.npy' % t.strftime('%Y%m%d'), np.array(Y))
        np.save(r'pictures/stock_%s.npy' % t.strftime('%Y%m%d'), np.array(asset))
