"""
Title: main.py

Created on: 2/17/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


plt.style.use('dark_background')
BATCH_SIZE = 64
EPOCHS = 10


def generate_sequences_from_df(dataframe: pd.DataFrame, seq_len: int = 30, predictions: int = 1) -> np.array:
    """ Generate a multidimensional array with dimensions:
        (BatchSize, SeqLen, SeqDim)
    """
    scaler = MinMaxScaler()
    data = dataframe.close.values
    output = []
    for i in range(len(data) - seq_len):
        seq = data[i: (seq_len+i)+predictions]
        seq = seq.reshape(-1, 1)
        scaler.fit(seq)
        seq = scaler.transform(seq)
        output.append(seq)
    return np.array(output)


def build_lstm_model():
    """ Build a simple neural network for a baseline evaluation. """
    model = models.Sequential([
        layers.LSTM(100, return_sequences=True, input_shape=(30, 1)),
        layers.LSTM(100, return_sequences=True),
        layers.LSTM(100, return_sequences=True),
        layers.TimeDistributed(layers.Dense(1, activation='linear'))
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.1),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError()]
    )
    return model


def build_dummy_model():
    """ Build a simple neural network for a baseline evaluation. """
    model = models.Sequential([
        layers.Flatten(input_shape=(30, 1)),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.1),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError()]
    )
    return model


def main():
    # Fetch raw data:
    dataset_url = 'https://storage.googleapis.com/open-ml-datasets/crypto/BitcoinPrices-20220216.csv'
    df = pd.read_csv(dataset_url)
    df['date_formatted'] = df.time.apply(lambda x: datetime.fromtimestamp(x).date())
    df = df.sort_values('time').reset_index(drop=True)
    df = df.loc[df.date_formatted != str(datetime.today().date())]
    df = df[['date_formatted', 'close']]

    # Scale data points between 0 & 1 and create datasets for training:
    sequences = generate_sequences_from_df(df)
    ds = tf.data.Dataset.from_tensor_slices(sequences)
    ds = ds.map(lambda x: (x[:30], x[-1]))
    total_seqs = len(ds)
    train_seqs = int(total_seqs * 0.8)
    X_train = ds.take(train_seqs).batch(BATCH_SIZE)
    X_test = ds.skip(train_seqs).batch(BATCH_SIZE)

    # Train a baseline model:
    baseline_model = build_dummy_model()
    baseline_model.fit(X_train, epochs=EPOCHS)
    predictions = baseline_model.predict(X_test)

    # Test the model:
    n = np.random.randint(0, BATCH_SIZE)
    test_seq, test_label = [x for x in X_test.as_numpy_iterator()][0]
    plt.figure(figsize=(13, 8))
    plt.plot(test_seq[n])
    plt.plot(30, test_label[n], 'gx')
    plt.plot(30, baseline_model.predict(test_seq[n].reshape((1, 30, 1))), 'ro')
    plt.show()
    return {}


if __name__ == '__main__':
    main()
