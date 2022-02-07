"""
Title: malaria_classifier.py

Created on: 2/2/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""

from tqdm import tqdm
from pytz import timezone
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
import os
import shutil
import random


AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = os.path.join(os.getcwd(), 'Datasets', 'Train')
TEST_DIR = os.path.join(os.getcwd(), 'Datasets', 'Test')
LOGS_DIR = os.path.join(os.getcwd(), 'Logs')
BATCH_SIZE = 32
IM_HEIGHT = 256
IM_WIDTH = 256
EPOCHS = 50


def get_logs_dir(model_name: str):
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return os.path.join(LOGS_DIR, f"{model_name}_{run_id}")


def reset_folders():
    # Destroy and recreate directories:
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'Infected'))
    os.makedirs(os.path.join(TRAIN_DIR, 'Uninfected'))
    os.makedirs(os.path.join(TEST_DIR, 'Infected'))
    os.makedirs(os.path.join(TEST_DIR, 'Uninfected'))


def split_raw_dataset(test_split: float = 0.2, reset: bool = False):
    # Check if work is already done:
    if reset is False:
        return None
    else:
        # Check total number of images available:
        infected = os.listdir(os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized'))
        uninfected = os.listdir(os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected'))
        total_images = len(infected) + len(uninfected)
        total_test_images = int(total_images * test_split)

        # Randomly select images from the raw dataset:
        random.seed(420)
        random.shuffle(infected)
        train_infected = infected[total_test_images:]
        train_uninfected = uninfected[total_test_images:]
        test_infected = infected[:total_test_images]
        test_uninfected = uninfected[:total_test_images]

        # Build directories:
        reset_folders()
        for img in tqdm(train_infected, desc='Building Training Infected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized', img)
            destination = os.path.join(TRAIN_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(train_uninfected, desc='Building Training Uninfected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected', img)
            destination = os.path.join(TRAIN_DIR, 'Uninfected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_infected, desc='Building Test Infected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Parasitized', img)
            destination = os.path.join(TEST_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_uninfected, desc='Building Test Uninfected folder'):
            source = os.path.join(os.getcwd(), 'Datasets', 'Raw', 'Uninfected', img)
            destination = os.path.join(TEST_DIR, 'Uninfected', img)
            shutil.copy(source, destination)


def get_dataset(batch_size: int, im_height: int, im_width: int, subset: str = 'training', validation: float = 0.2):
    return image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=validation,
        subset=subset,
        seed=420,
        image_size=(im_height, im_width),
        batch_size=batch_size,
    )


def get_baseline_nn():
    model_layers = [
        layers.Flatten(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
    model = Sequential(model_layers, name='BaselineModel')
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_nn():
    model_layers = [
        layers.Flatten(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ]
    model = Sequential(model_layers, name='CustomModelOne')
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_best_cnn():
    model_layers = [
        layers.Conv2D(64, 3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
    model = Sequential(model_layers, name='CustomModel-BestCNNSoFar')
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_cnn():
    model_layers = [
        layers.RandomFlip(),
        layers.RandomRotation(0.45),
        layers.Conv2D(64, 3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
    model = Sequential(model_layers, name='CustomModel-CNN')
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def main():
    # Prepare the raw data to be loaded:
    split_raw_dataset()

    # Load data into a generator:
    train_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)
    val_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, subset='validation')

    # Normalize data prior training:
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache the datasets:
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Start training:
    model = get_custom_cnn()
    tb_logs = TensorBoard(get_logs_dir(model.name))
    early_stop = EarlyStopping(patience=10)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop])
    return {}


if __name__ == "__main__":
    main()
