import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import pandas as pd

def load_data(directory, df):
    images = []
    labels = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(subdir, file)
            image = Image.open(path)
            image = image.resize((224, 224))
            image = np.array(image).astype('float64')
            if len(image.shape) == 2:
                # handle grayscale images
                image = np.stack((image,)*3, axis=-1).astype('float64')
            images.append(image)
            img_id = int(file.split('.')[0])
            label = df[df['id']==img_id]['label'].item() -1
            labels.append(label)
    return np.array(images), np.array(labels)

def train_model():
    # train = pd.read_csv('train_info.csv')
    # test = pd.read_csv('test_info.csv') 
    # train_dir = 'train_data'
    # test_dir = 'test_data'
    # X_train, y_train = load_data(train_dir, train)
    # X_test, y_test = load_data(test_dir, test)
    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test)
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    model.summary()
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    batch_size = 32
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)
    history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=10, 
                    validation_data=(X_test, y_test),
                    callbacks=[cp_callback])

# def run_code():


if __name__ == '__main__':
    train_model()
