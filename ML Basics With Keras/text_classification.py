"""
Title: text_classification.py

Created on: 1/6/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Binary classifier to perform sentiment analysis on an IMDB dataset
"""


from tensorflow.keras.utils import get_file, text_dataset_from_directory
from tensorflow.keras.layers import TextVectorization, Embedding, Dropout, GlobalAveragePooling1D, Dense, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import re
import os
import shutil
import string


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


def get_train_val_dataset(subset: str, directory: str, batch_size: int = 16, validation_split: float = 0.2):
    """ Since there's no validation set, let's create one from the training data. """
    seed = 69
    return text_dataset_from_directory(
        directory,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        subset=subset
    )


def get_test_dataset(directory: str, batch_size: int = 16):
    """ Since there's no validation set, let's create one from the training data. """
    return text_dataset_from_directory(
        directory,
        batch_size=batch_size
    )


def text_normalization(data):
    """ Clean text. """
    lowercase = tf.strings.lower(data)
    html_stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(html_stripped, '[%s]' % re.escape(string.punctuation), '')


def vectorized_layer(max_features: int, sequence_length: int):
    """ Create a TextVectorization layer to transform each word into a unique integer in an index. """
    return TextVectorization(
        standardize=text_normalization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )


def text_to_vector(vxt_layer, sample_text, sample_label):
    """" Apply processing layer to input text. """
    sample_text = tf.expand_dims(sample_text, -1)
    return vxt_layer(sample_text), sample_label


def build_neural_network(max_features: int, embedding_dim: int) -> Sequential:
    model = Sequential([
        Embedding(max_features + 1, embedding_dim),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(),
        metrics=BinaryAccuracy(threshold=0.0)
    )
    return model


def build_custom_network() -> Sequential:
    embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
    model = Sequential([
        hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True),
        Dropout(0.2),
        Dense(16),
        Dropout(0.2),
        Reshape((16, 1)),
        GlobalAveragePooling1D(),
        Dense(1)
    ])
    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(),
        metrics=BinaryAccuracy(threshold=0.0)
    )
    return model


def plot_loss(history_dict: dict):
    acc = history_dict['binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history_dict: dict):
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


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

    # Vectorize the datasets:
    MAX_FEATURES = 10_000
    SEQ_LENGTH = 420
    vxt_layer = vectorized_layer(max_features=MAX_FEATURES, sequence_length=SEQ_LENGTH)
    vxt_layer.adapt(raw_train_ds.map(lambda x, y: x))
    train_ds = raw_train_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))
    val_ds = raw_val_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))
    test_ds = raw_test_ds.map(lambda x, y: text_to_vector(vxt_layer=vxt_layer, sample_text=x, sample_label=y))

    # Optimize datasets:
    AUTOTUNE = tf.data.AUTOTUNE
    raw_train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    raw_val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build a network and train it:
    EMBEDDING_DIM = 32
    EPOCHS = 10
    # model = build_neural_network(max_features=MAX_FEATURES, embedding_dim=EMBEDDING_DIM)
    model = build_custom_network()
    training_data = model.fit(x=raw_train_ds, validation_data=raw_val_ds, epochs=EPOCHS)
    history_dict = training_data.history
    plot_loss(history_dict)
    plot_accuracy(history_dict)
    return {}


if __name__ == '__main__':
    main()
