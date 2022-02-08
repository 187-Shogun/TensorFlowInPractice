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
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import models
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import seaborn as sn
import pandas as pd
import os
import shutil
import random


AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = os.path.join(os.getcwd(), 'Datasets', 'Train')
TEST_DIR = os.path.join(os.getcwd(), 'Datasets', 'Test')
LOGS_DIR = os.path.join(os.getcwd(), 'Logs')
CM_DIR = os.path.join(os.getcwd(), 'ConfusionMatrixes')
MODELS_DIR = os.path.join(os.getcwd(), 'Models')
BATCH_SIZE = 32
IM_HEIGHT = 200
IM_WIDTH = 200
EPOCHS = 100
PATIENCE = 20
RANDOM_SEED = 420


def get_model_version_name(model_name: str):
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


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
        random.seed(RANDOM_SEED)
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
        seed=RANDOM_SEED,
        image_size=(im_height, im_width),
        batch_size=batch_size,
    )


def get_test_dataset(batch_size: int, im_height: int, im_width: int):
    return image_dataset_from_directory(
        TEST_DIR,
        seed=RANDOM_SEED,
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
    model = models.Sequential(model_layers, name='BaselineModel')
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
    model = models.Sequential(model_layers, name='CustomModelOne')
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
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
    model = models.Sequential(model_layers, name='CustomModel-BestCNNSoFar')
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_cnn():
    model_layers = [
        layers.RandomFlip(),
        layers.RandomRotation(0.3),
        layers.RandomContrast(0.3),
        layers.Conv2D(128, 3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dropout(0.2),
        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(1, activation='sigmoid')
    ]
    model = models.Sequential(model_layers, name='CustomModel-CNN')
    model.compile(
        optimizer=SGD(momentum=0.9, nesterov=True),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def plot_confision_matrix(model, test_dataset, version_name):
    # Fetch predictions and true labels:
    print("Fetching predictions...")
    predictions = []
    labels = []
    for x, y in test_dataset:
        predictions += list(model.predict(x).reshape(-1))
        labels += list(y.numpy().astype(float))

    # Build a confusion matrix and save the plot in a PNG file:
    matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
    df = pd.DataFrame(matrix)
    df.columns = test_dataset.class_names
    df.index = test_dataset.class_names
    cf = sn.heatmap(df, annot=True, fmt="d")
    cf.set(xlabel='Actuals', ylabel='Predicted')
    cf.get_figure().savefig(version_name)

    # Compute precision and recall:
    precision = metrics.Precision()
    precision.update_state(labels, predictions)
    print(f"Model's Precision: {precision.result().numpy()}")
    recall = metrics.Recall()
    recall.update_state(labels, predictions)
    print(f"Model's Recall: {recall.result().numpy()}")


def evaluate_existing_model():
    # Load test dataset and model from H5 file:
    test_ds = get_test_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)
    best_model = 'CustomModel-CNN_v.20220207-232113.h5'
    model = models.load_model(os.path.join(MODELS_DIR, best_model))

    # Fetch predictions and true labels:
    predictions = []
    labels = []
    for x, y in tqdm(test_ds, desc="Fetching predictions"):
        predictions += list(model.predict(x).reshape(-1))
        labels += list(y.numpy().astype(float))

    # Compute accuracy, precision and recall:
    print("Evaluating model on test data:")
    test_score = model.evaluate(test_ds)
    print(f"Model's Accuracy: {test_score}")
    precision = metrics.Precision()
    precision.update_state(labels, predictions)
    print(f"Model's Precision: {precision.result().numpy()}")
    recall = metrics.Recall()
    recall.update_state(labels, predictions)
    print(f"Model's Recall: {recall.result().numpy()}")


def main():
    # Prepare the raw data to be loaded:
    split_raw_dataset()

    # Load data into a generator:
    train_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)
    val_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, subset='validation')
    test_ds = get_test_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)

    # Normalize data prior training:
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache the datasets:
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Start training a single model:
    model = get_custom_cnn()
    version_name = get_model_version_name(model.name)
    tb_logs = TensorBoard(os.path.join(LOGS_DIR, version_name))
    early_stop = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=.5, patience=5)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Evaluate single model:
    test_score = model.evaluate(test_ds)
    print(f"Test Score: {test_score}")
    plot_confision_matrix(model, test_ds, os.path.join(CM_DIR, f"{version_name}.png"))
    return {}


if __name__ == "__main__":
    main()
