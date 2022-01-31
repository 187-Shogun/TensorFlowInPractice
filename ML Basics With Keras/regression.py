"""
Title: regression.py

Created on: 1/12/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Predict fuel efficiency for different car models.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Normalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def get_dataset() -> pd.DataFrame:
    """ Get dataset from the web. """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    cols = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    return pd.read_csv(
        url,
        names=cols,
        na_values='?',
        comment='\t',
        sep=' ',
        skipinitialspace=True
    )


def plot_data_distribution(df: pd.DataFrame):
    return sns.pairplot(df[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')


def get_normalization_layer(data: np.array) -> Normalization:
    """ Build a normalization layer and adapt it to the features in the dataset. """
    layer = Normalization(axis=-1, input_shape=[10])
    layer.adapt(data)
    return layer


def build_neural_network(normalization_layer: Normalization) -> Sequential:
    """ Build a simple neural network. """
    model = Sequential([
        normalization_layer,
        Dense(10, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    return model


def plot_loss(history_dict: dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history_dict: dict):
    acc = history_dict['mean_absolute_error']
    val_acc = history_dict['val_mean_absolute_error']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training MSE')
    plt.plot(epochs, val_acc, 'b', label='Validation MSE')
    plt.title('Training and validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def test_model(test_labels: np.array, test_predictions: np.array):
    plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()


def main():
    """ Run script. """
    # Clean raw data:
    df = get_dataset()
    avg_hp_by_cylinder = df.groupby(['Cylinders']).Horsepower.mean()
    avg_hp_by_cylinder.name = 'avg_hp_by_cylinder'
    df = df.join(avg_hp_by_cylinder, on='Cylinders')
    df.loc[df.Horsepower.isna(), 'Horsepower'] = df.loc[df.Horsepower.isna(), 'avg_hp_by_cylinder']
    df.Origin = df.Origin.map({1: "USA", 2: "Europe", 3: "Japan"})
    df = pd.get_dummies(df, columns=['Origin'], prefix='', prefix_sep='')

    # Split data into Train/Test sets:
    train_df = df.sample(frac=0.8, random_state=69)
    test_df = df.drop(train_df.index)

    # Separate labales from features:
    train_labels = train_df.pop('MPG')
    test_labels = test_df.pop('MPG')

    # Convert dataframes into arrays:
    train_labels = train_labels.values
    test_labels = test_labels.values
    train_df = train_df.values
    test_df = test_df.values

    # Build model and start training:
    EPOCHS = 100
    normalization_layer = get_normalization_layer(train_df)
    model = build_neural_network(normalization_layer)
    training_history = model.fit(x=train_df, y=train_labels, epochs=EPOCHS, validation_split=0.2)
    plot_loss(training_history.history)
    plot_accuracy(training_history.history)

    # Fetch some predictions to test the model:
    predictions = model.predict(test_df).flatten()
    test_model(test_labels, predictions)
    return {}


if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)
    main()
