import tensorflow as tf
import cv2
import numpy as np
from google.protobuf import text_format
import pandas as pd

import os
os.chdir('CUB_200_2011')
os.path.abspath('.')

labels = pd.read_csv("image_class_labels.txt", header=None, sep=" ")
labels.columns = ["id", "label"]
train_test = pd.read_csv("train_test_split.txt", header=None, sep=" ")
train_test.columns = ["id", "is_train"]
images = pd.read_csv("images.txt", header=None, sep=" ")
images.columns = ["id", "name"]

bounding_boxes = pd.read_csv("bounding_boxes.txt", header=None, sep=" ")
bounding_boxes.columns = ["id", "x", "y","width", "height"]
images_bb = pd.merge(images, bounding_boxes, on=["id"])
images_bb_train_test = pd.merge(images_bb, train_test, on=["id"])

merged_df = pd.merge(images_bb_train_test, labels, on=["id"]) 
train = merged_df[merged_df['is_train']==1]
test = merged_df[merged_df['is_train']==0]
len(test)

for index, row in train.iterrows():
    image_name = f"{row['id']}.jpg"
    os.rename(f"images/{row['name']}", f'train_data/{image_name}')
    row['img_name'] = image_name
for index, row in test.iterrows():
    image_name = f"{row['id']}.jpg"
    os.rename(f"images/{row['name']}", f'test_data/{image_name}')
    row['img_name'] = image_name

test.head()
test.to_csv('test_info.csv', index=False)
train.to_csv('train_info.csv', index=False)