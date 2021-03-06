"""
Title: main.py

Created on: 1/6/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Binary classifier to perform sentiment analysis on an IMDB dataset
"""


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from datetime import datetime
from pytz import timezone
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text
import os
import re
import string

AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 10
RANDOM_SEED = 69
MAX_FEATURES = 25_000
SEQ_LENGTH = 128
EMBEDDING_DIM = 64
LOGS_DIR = os.path.join(os.getcwd(), 'Logs')
CM_DIR = os.path.join(os.getcwd(), 'ConfusionMatrixes')
MODELS_DIR = os.path.join(os.getcwd(), 'Models')


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def get_dataset():
    """ Get dataset from the TFDS library. """
    train_a, info = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        split=['train'],
        batch_size=BATCH_SIZE
    )
    a, b, c = tfds.even_splits('test', n=3, drop_remainder=True)
    train_b, val, test = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=[a, b, c],
        batch_size=BATCH_SIZE
    )
    # Unpack elements:
    train = train_a[0].concatenate(train_b)
    return train, val, test, info


def text_normalization(data):
    """ Clean text. """
    lowercase = tf.strings.lower(data)
    html_stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(html_stripped, '[%s]' % re.escape(string.punctuation), '')


def txt_to_vec_layer(max_features: int, sequence_length: int):
    """ Create a TextVectorization layer to transform each word into a unique integer in an index. """
    corpus_a = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=['train'],
    )
    corpus_b = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=['unsupervised'],
    )
    vxt = layers.TextVectorization(
        standardize=text_normalization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vxt.adapt(corpus_a[0].concatenate(corpus_b[0]).map(lambda x, y: x))
    return vxt


def text_to_vector(vxt_layer, sample_text, sample_label):
    """" Apply processing layer to input text. """
    sample_text = tf.expand_dims(sample_text, -1)
    return vxt_layer(sample_text), sample_label


def build_dummy_network() -> Sequential:
    """ Build a baseline network. """
    lyrs = [
        txt_to_vec_layer(max_features=MAX_FEATURES, sequence_length=SEQ_LENGTH),
        layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
        layers.GlobalMaxPool1D(),
        layers.Dropout(0.3),
        layers.Dense(1)
    ]
    model = Sequential(name='Dummy-NN', layers=lyrs)
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy()
    )
    return model


def build_custom_network() -> Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ]
    model = Sequential(name='Pretrained-NN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy()
    )
    return model


# noinspection PyCallingNonCallable
def bert_network() -> tf.keras.Model:
    """ Return a pretrained BERT model from tensorflow hub. """
    # TFHub Sources:
    preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"

    # Assemble model:
    text_input = layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(preprocessor_url)
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(encoder_url, trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]
    cls_layer = layers.Dense(1)
    cls_output = cls_layer(pooled_output)

    # Compile it and return it:
    model = tf.keras.Model(inputs=text_input, outputs=cls_output, name='BERT-DNN')
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy()
    )
    return model


def train_pretrained_network(training_ds, validation_ds, pretrain_rounds=10):
    """ Build a model from a pretrained one. Freeze the bottom layers and train the top layers
    for n epochs. Then, unfreeze the bottom layers, adjust the learning rate down and train the
    model one more time. """
    # Start pretraining:
    model = build_dummy_network()
    version_name = get_model_version_name(model.name)
    tb_logs = callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    model.fit(training_ds, validation_data=validation_ds, epochs=pretrain_rounds, callbacks=[tb_logs])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Unfreeze layers and train the entire network:
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy(threshold=0.0)
    )
    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def main():
    """ Run script. """
    # Fetch datasets and configure them:
    X_train, X_val, X_test, info = get_dataset()
    X_train = X_train.cache().prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)

    # Build a network and train it:
    model = bert_network()
    model.fit(X_train, validation_data=X_val, epochs=EPOCHS)
    score = model.evaluate(X_test)
    print(score)
    return {}


if __name__ == '__main__':
    main()
