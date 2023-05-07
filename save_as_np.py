import numpy as np
import os
from PIL import Image
import pandas as pd
import tensorflow as tf

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


def read_and_persist():
    train = pd.read_csv('train_info.csv')
    test = pd.read_csv('test_info.csv') 
    train_dir = 'train_data'
    test_dir = 'test_data'
    X_train, y_train = load_data(train_dir, train)
    X_test, y_test = load_data(test_dir, test)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    np.save('train_images.npy', X_train)
    np.save('train_labels.npy', y_train)
    np.save('test_images.npy', X_test)
    np.save('test_labels.npy', y_test)

if __name__ == '__main__':
    load_data()