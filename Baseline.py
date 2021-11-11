import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from BackTest import BackTest

class baselineModel:
    # baseline model
    # Feature Extraction Layer + Dense Layer

    def __init__(self, name, config, fit_config):
        self.name = name
        self.config = config
        self.fit_config = fit_config

        tf.random.set_seed(1)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(270,)),
            tf.keras.layers.BatchNormalization(),  # normalize data

            tf.keras.layers.Dense(units=90, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dropout(rate=0.5),  # drop out 50% of the neurons
            # tf.keras.layers.BatchNormalization(),  # normalize data
            tf.keras.layers.Dense(units=30, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dropout(rate=0.5),  # drop out 50% of the neurons
            # tf.keras.layers.BatchNormalization(),  # normalize data
            tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.TruncatedNormal())
        ])
        opt = tf.keras.optimizers.Adam(lr=self.config['learning_rate'], clipvalue=0.5)
        self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])

    def fit(self, x, y):
        self.model.fit(
            x=x,
            y=y,
            **self.fit_config
        )

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred



def baselineResult(begt, endt):
    config = {
        'model_path': Path('models'),
        'feat_num': 270,
        'learning_rate': 0.002,
        'loss': 'mse',
        'metrics': 'mse',
    }

    fit_config = {
        'batch_size': 3000,
        'epochs': 100,
    }
    data = pd.read_parquet("data.parquet")
    date_list = data.date.unique()
    beg_n = np.argmax(date_list[date_list <= pd.to_datetime(begt)]) + 1
    end_n = np.argmax(date_list[date_list <= pd.to_datetime(endt)])
    df_predict = pd.DataFrame(columns=['date','stock','score'])
    for n in np.arange(beg_n, end_n, 10):
        x_train_raw = []
        y_train = []
        for i in np.arange(n-1,n-201,-5):
            t = pd.to_datetime(date_list[i]).strftime('%Y%m%d')
            X_t = np.load("pictures/X_%s.npy" % t)
            Y_t = np.load("pictures/Y_%s.npy" % t)
            if len(x_train_raw):
                x_train_raw = np.concatenate((x_train_raw, X_t), axis=0)
            else:
                x_train_raw = X_t
            if len(y_train):
                y_train = np.concatenate((y_train, Y_t), axis=0)
            else:
                y_train = Y_t
        x_train_raw = x_train_raw.reshape(x_train_raw.shape[0], 270)

        selector = baselineModel('stock_selector_on_%s' % t, config, fit_config)
        selector.fit(x_train_raw, y_train)

        predict_t = pd.to_datetime(date_list[n]).strftime('%Y%m%d')
        x_predict = np.load("pictures/X_%s.npy" % predict_t)
        x_predict = x_predict.reshape(x_predict.shape[0], 270)
        stock_predict = np.load("pictures/stock_%s.npy" % predict_t)
        y_predict = selector.predict(np.asarray(x_predict).astype(np.float32))

        score_t = pd.DataFrame(np.transpose([stock_predict,y_predict.reshape(-1)]),columns=['stock','score'])
        score_t['date'] = pd.to_datetime(date_list[n])

        df_predict = pd.concat([df_predict,score_t])
        print (t)
    df_predict = df_predict.set_index(['date','stock'])
    df_predict.to_parquet("predictions/%s-%s.parquet"%(begt,endt))

    bt = BackTest(df_predict)
    bt.backtest()

    print(bt.netvalue)
    print(bt.evaluation())

    return bt