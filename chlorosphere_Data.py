import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Define Dataset Paths
data_dir = r"C:\Users\ecarl\OneDrive\Desktop\chlorosphere_Proj2\color"

#Validate dataset directory
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The dataset directory {data_dir} does not exist.")
if not os.listdir(data_dir):
    raise FileNotFoundError(f"The dataset directory {data_dir} is empty.")

#Definefunctions for preprocessing
def define_paths_and_labels(data_dir):

    filepaths = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            for file in os.listdir(folder_path):
                if file.lower().endswith(valid_extensions):  # Check valid image extensions
                    filepaths.append(os.path.join(folder_path, file))
                    labels.append(folder)  # Use folder name as label
    return filepaths, labels

#Collects file paths and labels
files, classes = define_paths_and_labels(data_dir)
df = pd.DataFrame({"filepaths": files, "labels": classes})

#Makes sure labels are strings
df['labels'] = df['labels'].astype(str)

#Split Dataset
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['labels'], random_state=123
)
valid_df, test_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df['labels'], random_state=123
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")
print(f"Testing samples: {len(test_df)}")

#Data augmentation and generators
img_size = (224, 224)
batch_size = 32

#Training data generator with augmentation
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

#Validation and testing data generator without augmentation
valid_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)

#Prepare data generators
train_generator = train_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size,
    class_mode='categorical', batch_size=batch_size, shuffle=True
)

valid_generator = valid_gen.flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
    class_mode='categorical', batch_size=batch_size, shuffle=False
)

test_generator = test_gen.flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels', target_size=img_size,
    class_mode='categorical', batch_size=batch_size, shuffle=False
)

# Debugging the generator output
data_batch, label_batch = next(iter(train_generator))
print("Data batch shape:", data_batch.shape)  
print("Label batch shape:", label_batch.shape) 
