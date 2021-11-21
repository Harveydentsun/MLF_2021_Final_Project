import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from backtest import BackTest
from tqdm import tqdm


class CNNModel:
    """
    CNN Model
    ------------------
    Input Layer:
        702 * 1 extracted features, batch normalized, coming from CNN feature extraction

    1 Hidden Dense Layers:
        30 units with ReLU activation function, 50% dropout

    Output Layer:
        1 unit with Linear activation function
    -------------------
    """

    def __init__(self, name, config, fit_config):
        self.name = name
        self.config = config
        self.fit_config = fit_config

        tf.random.set_seed(1)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.config['feat_num'],)),
            tf.keras.layers.BatchNormalization(),  # normalize data

            tf.keras.layers.Dense(units=30, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dropout(rate=0.5),  # drop out 50% of the neurons
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal())
        ])
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=10,
                                                     verbose=0, mode="min", baseline=None,
                                                     restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'], monitor='loss', verbose=0,
                                                        save_best_only=True, mode='min')
        self.cb_list = [earlystop, checkpoint]
        opt = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'], clipvalue=0.5)
        self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])

    def fit(self, x, y):
        self.model.fit(
            x=x,
            y=y,
            callbacks=self.cb_list,
            **self.fit_config
        )

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred


def cnn_result(begt, endt):
    """
    Calculate CNN model backtest result
    ----------------
    :param begin_t: str
        backtest begin date

    :param end_t: str
        backtest end date

    :return: bt: class<BackTest>
        backtest results
    ----------------
    """
    # Model config
    config = {
        'model_path': Path('models'),
        'feat_num': 702,
        'learning_rate': 0.004,
        'loss': 'mse',
        'metrics': 'mse',
    }

    fit_config = {
        'batch_size': 2000,
        'epochs': 10000,
    }
    # read data
    data = pd.read_parquet("data.parquet")
    # get all trade days
    date_list = data.date.unique()
    beg_n = np.argmax(date_list[date_list <= pd.to_datetime(begt)]) + 1
    end_n = np.argmax(date_list[date_list <= pd.to_datetime(endt)])
    # predicting results
    df_predict = pd.DataFrame(columns=['date', 'stock', 'score'])
    sig = True
    for n in tqdm(np.arange(beg_n, end_n, 10)):
        # Training inputs for CNN model, start from 12 days before.
        # The reason is that to trade in t, we can only use t-1 data to predict, then training
        # data should started from t-12 (with return label from t-11 to t-1)
        if sig:
            x_train_raw = []
            y_train = []
            for i in np.arange(n - 12, n - 512, -5):
                t = pd.to_datetime(date_list[i]).strftime('%Y%m%d')
                X_t = np.load("features/%s.npy" % t)
                Y_t = np.load("pictures/Y_%s.npy" % t)
                if len(x_train_raw):
                    x_train_raw = np.concatenate((x_train_raw, X_t), axis=0)
                else:
                    x_train_raw = X_t
                if len(y_train):
                    y_train = np.concatenate((y_train, Y_t), axis=0)
                else:
                    y_train = Y_t

            # delete nan and inf data
            isnum = ~ (np.isnan(x_train_raw).max(axis=1) | np.isinf(x_train_raw).max(axis=1))
            x_train_raw = x_train_raw[isnum]
            y_train = y_train[isnum]

            # shuffle data
            shuffle = np.random.permutation(x_train_raw.shape[0])
            x_train_raw = x_train_raw[shuffle]
            y_train = y_train[shuffle]

            # Model training
            selector = CNNModel('CNN_selector_on_%s' % t, config, fit_config)
            selector.fit(x_train_raw, y_train)
        sig = not sig

        # predicting today's label
        predict_t = pd.to_datetime(date_list[n - 1]).strftime('%Y%m%d')
        x_predict = np.load("features/%s.npy" % predict_t)
        stock_predict = np.load("pictures/stock_%s.npy" % predict_t)
        y_predict = selector.predict(np.asarray(x_predict).astype(np.float32))
        # combined with stock ticker
        score_t = pd.DataFrame(np.transpose([stock_predict, y_predict.reshape(-1)]), columns=['stock', 'score'])
        # convert data type and drop na prediction
        score_t.score = score_t.score.astype(float)
        score_t['date'] = pd.to_datetime(date_list[n])
        score_t = score_t.dropna()

        df_predict = pd.concat([df_predict, score_t])
        print(date_list[n])

    # save all predictions to /predictions as .parquet file
    df_predict = df_predict.set_index(['date', 'stock'])
    df_predict.to_parquet("predictions/CNN_%s_%s.parquet" % (begt, endt))
    # Backtest based on model prediction
    bt = BackTest(df_predict)
    bt.backtest()

    print(bt.netvalue)
    print(bt.evaluation())

    return bt


if __name__ == "__main__":
    bt = cnn_result('2018-01-01', '2020-12-31')
