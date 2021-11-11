import datetime
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import pickle


class CNN_Model:
    # CNN_Model
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


# %%
# configuration
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

# load and prepare data for training and validating
dir_ = Path("data")
with open(dir_ / 'x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open(dir_ / "y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

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
date_str = datetime.datetime.today().strftime("%Y%m%d-%H%M")
selector_to_train = CNN_Model('stock_selector_on_%s' % date_str, config, fit_config)
selector_to_train.fit(x_train2, y_train2, validation_data=(x_val, y_val))
selector_to_train.save('models/', selector_to_train.name)


# %%
