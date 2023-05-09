import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from meta_layers import MetaConv2D



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


def get_resnet_model():
    resnet_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in resnet_model.layers[:-10]:
        layer.trainable = False
    # Add new fully connected layers
    x = resnet_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(200, activation='softmax')(x)
    model = tf.keras.models.Model(resnet_model.input, x)
    print("returning resnet50 model")
    return model


def train_model():
    X_train, y_train, X_test, y_test = load_saved_data()
    
    # Data augmentation
    data_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest')
    
    # Load the pre-trained ResNet50V2 model
    base_model = ResNet50V2(input_shape=(224, 224, 3),
                            include_top=False,
                            weights='imagenet')
    
    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Define the MetaConv2D layer
    x = MetaConv2D(128, (3, 3))(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    
    # Define the dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Define the output layer
    output_layer = layers.Dense(200, activation='softmax')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
    
    # Define the checkpoints
    checkpoint_path = "resnet_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)
    
    # Train the model
    history = model.fit(data_augmentation.flow(X_train, y_train, batch_size=32),
                        epochs=150, 
                        validation_data=(X_test, y_test),
                        callbacks=[cp_callback], validation_freq=10)
    
    # Save the model
    model.save('resnet50_model.h5')


if __name__ == '__main__':
    train_model()

