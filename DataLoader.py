import datetime

import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import feature_extraction_file as feature_extraction



def compute_feature_data(t):
    data = pd.read_parquet(r"data.parquet")
    date_list = data.date.unique()
    n = np.argmax(date_list[date_list < pd.to_datetime(t)])
    x_train_raw = []
    y_train = []
    for i in np.arange(n, n - 200, -5):
        t = pd.to_datetime(date_list[i]).strftime('%Y%m%d')
        X_t = np.load("pictures/X_%s.npy" % t)
        Y_t = np.load("pictures/Y_%s.npy" % t)
        if len(x_train_raw):
            x_train_raw = np.concatenate((x_train_raw,X_t),axis=0)
        else:
            x_train_raw = X_t
        if len(y_train):
            y_train = np.concatenate((y_train,Y_t),axis=0)
        else:
            y_train = Y_t

    x_train = feature_extraction.pipeline(x_train_raw)
    return x_train, y_train


class StockSelectorAlpha:
    # StockSelector Alpha
    # Feature Extraction Layer + Dense Layer

    def __init__(self, name, config, fit_config):
        self.name = name
        self.config = config
        self.fit_config = fit_config

        tf.random.set_seed(1)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(351,)),
            tf.keras.layers.Dropout(rate=0.2),  # drop out 20% of the input
            tf.keras.layers.BatchNormalization(),  # normalize data

            tf.keras.layers.Dense(units=64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dropout(rate=0.5),  # drop out 50% of the neurons
            # tf.keras.layers.BatchNormalization(),  # normalize data
            tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.TruncatedNormal())
        ])
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=33,
                                                     verbose=0, mode="min", baseline=None,
                                                     restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'], monitor='val_loss', verbose=0,
                                                        save_best_only=True, mode='min')
        self.cb_list = [earlystop, checkpoint]
        opt = tf.keras.optimizers.Adam(lr=self.config['learning_rate'], clipvalue=0.5)
        self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])

    def fit(self, x, y, validation_data):
        self.model.fit(
            x=x,
            y=y,
            validation_data=(validation_data),
            callbacks=self.cb_list,
            **self.fit_config
        )

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def save(self, path, name):
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save(path + name + '.h5')
        pd.DataFrame(self.model.history.history).to_csv(path + name + '_train_history.csv')
        pd.DataFrame(self.model.history.history)[['loss', 'val_loss']].plot()
        plt.title(f"{self.name}")
        plt.show()
        plt.savefig(path + name + '_train_history.png')



if __name__ == "__main__":
    config = {
        'model_path': Path('models'),
        'feat_num': 351,
        'learning_rate': 0.002,
        'loss': 'mse',
        'metrics': 'mse',
    }

    fit_config = {
        'batch_size': 2560,
        'epochs': 10000,
    }
    for t in ['2019-12-31', '2020-02-06','2020-03-05', '2020-04-02',
       '2020-05-06', '2020-06-03','2020-07-03', '2020-07-31',
       '2020-08-28', '2020-09-25','2020-11-02', '2020-11-30']:

        # load and prepare data for training and validating
        x_train, y_train = compute_feature_data(t)
        with open("data/x_train.pkl", "rb") as f:
            # x_train = pickle.load(f)
            pickle.dump(np.array(x_train), f)
            f.close()
        with open("data/y_train.pkl", "rb") as f:
            # y_train = pickle.load(f)
            pickle.dump(np.array(y_train), f)
            f.close()

        _train_data = pd.DataFrame(
            np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)).dropna()  # drop na values in the X and Y
        _train_data = _train_data.sample(frac=1, random_state=1)
        thr = round(_train_data.shape[0] * 0.75)
        train_data = _train_data.iloc[:thr, :]
        x_train2 = np.array(train_data.iloc[:, :351])
        y_train2 = np.array(train_data.iloc[:, 351]).reshape(-1, 1)

        val_data = _train_data.iloc[thr:, :]
        x_val, y_val = np.array(val_data.iloc[:, :351]), np.array(val_data.iloc[:, 351]).reshape(-1, 1)
        # %%
        # Train model
        date_str = t
        selector_to_train = CNN_Model('stock_selector_on_%s' % date_str, config, fit_config)
        selector_to_train.fit(x_train2, y_train2, validation_data=(x_val, y_val))
        selector_to_train.save('models/', selector_to_train.name)

