import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


class LSTM_Model:
    # LSTM model
    # LSTM Layer + Dense Layer

    def __init__(self, name, config, fit_config):
        self.name = "lstm_" + name
        self.config = config
        self.fit_config = fit_config

        tf.random.set_seed(1)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(9, 30)),
            tf.keras.layers.Dropout(rate=0.2),    # drop out 20% of the input
            tf.keras.layers.BatchNormalization(),  # normalize data
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(64, return_sequences=False, recurrent_dropout=0.5),   # drop out 50% of the input
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.TruncatedNormal())
        ])
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=33,
                                                     verbose=0, mode="min", baseline=None,
                                                     restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'], monitor='val_loss', verbose=0,
                                                        save_best_only=True, mode='min')
        self.cb_list = [earlystop, checkpoint]
        opt = tf.keras.optimizers.Adam(lr=self.config['learning_rate'], clipvalue=0.5)

        if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            self.GPU = False
            self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])
        else:
            self.GPU = True
            self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])

    def fit(self, x, y, validation_data=None):
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




# %%
# train stocks
config = {
    'model_path': Path('models'),
    'learning_rate': 0.002,
    'loss': 'mse',
    'metrics': 'mse',
}

fit_config = {
    'batch_size': 2560,
    'epochs': 10000,
}

# load training data
dir_ = Path("data")
with open(dir_ / 'x_train_raw.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open(dir_ / "y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
assert x_train.shape[0] == y_train.shape[0]

_train_data = pd.DataFrame(y_train).sample(frac=1, random_state=0)
thr = round(_train_data.shape[0] * 0.75)
id_train = list(_train_data.iloc[:thr, :].index)
x_train2 = x_train[id_train, :, :]
y_train2 = y_train[id_train]

id_val = list(_train_data.iloc[thr:, :].index)
x_val = x_train[id_val, :, :]
y_val = y_train[id_val]
assert x_val.shape[0] + x_train2.shape[0] == x_train.shape[0]

# %%
date_str = datetime.datetime.today().strftime("%Y%m%d-%H%M")
selector_to_train = LSTM_Model('stock_selector_on_%s' % date_str, config, fit_config)
selector_to_train.fit(x_train2, y_train2, validation_data=(x_val, y_val))
selector_to_train.save('models/', selector_to_train.name)
# %%

# %%
