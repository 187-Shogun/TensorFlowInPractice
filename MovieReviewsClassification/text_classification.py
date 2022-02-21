"""
Title: text_classification.py

Created on: 1/6/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Binary classifier to perform sentiment analysis on an IMDB dataset
"""


from tensorflow.keras.utils import get_file, text_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
import tensorflow as tf
import re
import os
import shutil
import string


# Configurations:
AUTOTUNE = tf.data.AUTOTUNE
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 10
MAX_FEATURES = 10_000
SEQ_LENGTH = 256


def remove_unnecesary_folders(dataset_path: str) -> None:
    """ Remove folder with additional unsupervised examples. """
    folder = 'unsup'
    path = os.path.join(dataset_path, folder)
    if os.path.exists(path):
        shutil.rmtree(path=path)
    else:
        pass


def get_imdb_dataset() -> str:
    """ Download and extract the dataset. """
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset_path = './Datasets/Cache/aclImdb'
    if os.path.exists(os.path.join(dataset_path, 'test')) and os.path.exists(os.path.join(dataset_path, 'train')):
        return dataset_path
    else:
        get_file('aclImdb_v1', url, untar=True, cache_dir='./Datasets/Cache', cache_subdir='')
        return dataset_path


def get_train_val_dataset(subset: str, directory: str, validation_split: float = 0.2):
    """ Since there's no validation set, let's create one from the training data. """
    seed = 69
    return text_dataset_from_directory(
        directory,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        seed=seed,
        subset=subset
    )


def get_test_dataset(directory: str):
    """ Since there's no validation set, let's create one from the training data. """
    return text_dataset_from_directory(
        directory,
        batch_size=BATCH_SIZE
    )


def text_normalization(data):
    """ Clean text. """
    lowercase = tf.strings.lower(data)
    html_stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(html_stripped, '[%s]' % re.escape(string.punctuation), '')


def vectorized_layer():
    """ Create a TextVectorization layer to transform each word into a unique integer in an index. """
    return layers.TextVectorization(
        standardize=text_normalization,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQ_LENGTH
    )


def text_to_vector(vxt_layer, sample_text, sample_label):
    """" Apply processing layer to input text. """
    sample_text = tf.expand_dims(sample_text, -1)
    return vxt_layer(sample_text), sample_label


def build_neural_network() -> Sequential:
    model = Sequential([
        layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
        layers.BatchNormalization(),
        layers.LSTM(EMBEDDING_DIM*2, return_sequences=True),
        layers.BatchNormalization(),
        layers.LSTM(EMBEDDING_DIM * 2, return_sequences=True),
        layers.BatchNormalization(),
        layers.LSTM(EMBEDDING_DIM*2),
        layers.Dense(1)
    ])
    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(),
        metrics=BinaryAccuracy(threshold=0.0)
    )
    return model


def build_dummy_network() -> Sequential:
    model = Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(),
        metrics=BinaryAccuracy(threshold=0.0)
    )
    return model


def main():
    """ Run script. """
    # Fetch datasets:
    dataset_path = get_imdb_dataset()
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    remove_unnecesary_folders(train_dir)

    # Load them into a TF generator:
    raw_train_ds = get_train_val_dataset(subset='training', directory=train_dir)
    raw_val_ds = get_train_val_dataset(subset='validation', directory=train_dir)
    raw_test_ds = get_test_dataset(directory=test_dir)

    # Optimize datasets:
    vxt_layer = vectorized_layer()
    vxt_layer.adapt(raw_train_ds.map(lambda x, y: x))
    X_train = raw_train_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))
    X_val = raw_val_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))
    X_test = raw_test_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))

    X_train = X_train.cache().prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)

    # Build a network and train it:
    model = build_neural_network()
    model.fit(x=X_train, validation_data=X_val, epochs=EPOCHS)
    score = model.evaluate(X_test)
    print(score)
    return {}


if __name__ == '__main__':
    main()
