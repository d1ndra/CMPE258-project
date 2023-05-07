import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

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


def load_saved_data(train_im="train_images.npy", train_l="train_labels.npy", test_im="test_images.npy", test_l="test_labels.npy"):
    print("Init Loading data")
    X_train = np.load(train_im)
    y_train = np.load(train_l)
    X_test = np.load(test_im)
    y_test = np.load(test_l)
    print("Finished Loading data")
    return X_train, y_train, X_test, y_test

def get_mobile_netv2():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    head = layers.Conv2D(4, kernel_size=3, padding='same')(base_model.output)
    head = layers.Flatten()(head)
    model = tf.keras.Model(inputs=base_model.input, outputs=head)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='softmax')
    ])
    print("returning mobilenet model")
    return model

def get_resnet_model():
    resnet_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in resnet_model.layers[:-10]:
        layer.trainable = False
    # Add new fully connected layers
    x = tf.keras.layers.Flatten()(resnet_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(200, activation='softmax')(x)
    model = tf.keras.models.Model(resnet_model.input, x)
    print("returning resnet50 model")
    return model

def train_model():
    X_train, y_train, X_test, y_test = load_saved_data()
    model = get_resnet_model()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    checkpoint_path = "resnet_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    batch_size = 32
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)
    history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=200, 
                    validation_data=(X_test, y_test),
                    callbacks=[cp_callback])
    model.save('resnet50_model.h5')

if __name__ == '__main__':
    train_model()
