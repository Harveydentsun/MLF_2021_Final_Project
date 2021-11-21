import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import bottleneck as bn
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed


class CalculatorType:
    """
    Calculator type for convolution and pooling function
    """
    SINGLE = 'Single_Variable'
    DUAL = 'Dual_Variable'


def rolling_window(a: np.array, window_size: int, step: int):
    """
    Convert 1-d array to a rolling window 2-d array
    ------------------
    :param a: np.array
        Array need to be converted

    :param window_size: int
        Number of members in a single rolling window

    :param step: int
        Index difference between two rolling window

    :return: 2-d np.array
    ------------------
    """
    shape = ((a.shape[0] - window_size) // step + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def cal_corr_cov(x, y, field='corr'):
    """Correlate / Covariant each x with each y.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape N X T.

    field: str
      corr or cov

    Returns
    -------
    np.array
      N X 1 array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                   mu_y[np.newaxis, :])
    if field == 'cov':
        return cov[0]
    elif field == 'corr':
        return (cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :]))[0]


class TSCorr:
    """
    Compute rolling correlation between two arrays
    """
    type_ = CalculatorType.DUAL

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray, y: np.ndarray):
        X, Y = rolling_window(x, self.window, self.step), rolling_window(y, self.window, self.step)
        return cal_corr_cov(X, Y, "corr")


class TSCov:
    """
    Compute rolling covariance between two arrays
    """
    type_ = CalculatorType.DUAL

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray, y: np.ndarray):
        X, Y = rolling_window(x, self.window, self.step), rolling_window(y, self.window, self.step)
        return cal_corr_cov(X, Y, "cov")


class TSStd:
    """
    Compute rolling standard deviation of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanstd(X, 1)


class TSZscore:
    """
    Compute rolling Z-score of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmean(X, 1) / bn.nanstd(X, 1)


class TSReturn:
    """
    Compute rolling return of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return X[:, -1] / X[:, 0] - 1


class TSDecaylinear:
    """
    Compute rolling decayed linear average of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        weight = np.arange(self.window, 0, -1)
        weight = weight / weight.sum()
        return X @ weight


class TSMin:
    """
    Compute rolling min of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmin(X, 1)


class TSMax:
    """
    Compute rolling max of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmax(X, 1)


class TSMean:
    """
    Compute rolling mean of an array
    """
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmean(X, 1)


def parallel_cal_single_col(data: np.array, row: int, calculator):
    """
    Apply a given single variable calculator to a row in every member of the input data
    """
    feature = [calculator.cal(x[row]) for x in data]
    return feature


def parallel_cal_dual_col(data: np.array, row1: int, row2: int, calculator):
    """
    Apply a given dual variable calculator to two given rows in every member of the input data
    """
    feature = [calculator.cal(x[row1], x[row2]) for x in data]
    return feature


def feature_extraction(data: np.array, config: dict):
    """
    Extract features using convolution and pooling
    --------------------
    :param data: 3-d np.array
        An numpy array containing all data pictures
        Should have a dimension of n*9*30

    :param config: dict
        Information of convolution/pooling procedures

    :return: 3-d np.array
        Augmented features
    --------------------
    """
    # calculator information
    augmentation_map = {
        "ts_corr": TSCorr,
        "ts_cov": TSCov,
        "ts_stddev": TSStd,
        "ts_zscore": TSZscore,
        "ts_return": TSReturn,
        "ts_decaylinear": TSDecaylinear,
        "ts_min": TSMin,
        "ts_max": TSMax,
        "ts_mean": TSMean
    }
    results = []
    # use parallel computing to calculate all convolutional pictures
    for calculator_name in config['calculator']:
        calculator = augmentation_map[calculator_name](config['window'], config['step'])
        if calculator.type_ == CalculatorType.SINGLE:
            results += Parallel(n_jobs=8)(delayed(parallel_cal_single_col)(data, row, calculator) for row in
                                          np.arange(data.shape[1]))

        elif calculator.type_ == CalculatorType.DUAL:
            results += Parallel(n_jobs=8)(delayed(parallel_cal_dual_col)(data, row1, row2, calculator) for
                                          row1, row2 in combinations(np.arange(data.shape[1]), 2))
    # combine all pictures
    df = np.hstack(results)
    # reshape as a 3-d array
    df = df.reshape([df.shape[0], 1, df.shape[1]])
    return df


def cal_all_features():
    """
    Calculate all features for all data pictures
    """
    # load all trade day
    data = pd.read_parquet("data.parquet")
    date_list = data.date.unique()

    # Convolution layer config
    config1 = {'calculator': ["ts_corr", "ts_cov", "ts_stddev",
                              "ts_zscore", "ts_return", "ts_decaylinear", "ts_mean"],
               "window": 10,
               "step": 10}

    # Pooling layer config
    config2 = {'calculator': ["ts_min", "ts_max", "ts_mean"],
               "window": 3,
               "step": 3}

    # calculate along all date
    for i, t in enumerate(tqdm(date_list)):
        if i < 29:
            continue
        # read data pictures of that day
        t_str = pd.to_datetime(t).strftime('%Y%m%d')
        X_t = np.load("pictures/X_%s.npy" % t_str)

        # convolution layer
        data = feature_extraction(X_t, config1)

        # pooling layer
        data2 = feature_extraction(data, config2)

        # reshape to 2-d array and stack two results
        data = data.reshape([data.shape[0],data.shape[2]])
        data2 = data2.reshape([data2.shape[0], data2.shape[2]])
        feature = np.hstack([data,data2])

        # save as .npy file under /features
        np.save(r'features/%s.npy' % t_str, feature)
