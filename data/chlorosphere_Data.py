import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Define Dataset Paths
original_data_dir = "data/colour"  # Path to original dataset
data_dir = "data/colour_resized"  # Path for resized dataset

# Define functions for preprocessing
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

# Collect file paths and labels
files, classes = define_paths_and_labels(data_dir)
df = pd.DataFrame({"filepaths": files, "labels": classes})
df['labels'] = df['labels'].astype(str)  # Ensure labels are strings

# Cache or Load Dataset Splits
try:
    train_df = pd.read_pickle("data/train_data.pkl")
    valid_df = pd.read_pickle("data/valid_data.pkl")
    test_df = pd.read_pickle("data/test_data.pkl")
    print("Loaded cached dataset splits.")

    # Save splits as CSV files
    train_df.to_csv("data/train_data.csv", index=False)
    valid_df.to_csv("data/valid_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)
    print("CSV files generated from existing .pkl files.")
except FileNotFoundError:
    print("Splitting dataset...")
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['labels'], random_state=123
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=1/3, stratify=temp_df['labels'], random_state=123
    )
    train_df.to_pickle("data/train_data.pkl")
    valid_df.to_pickle("data/valid_data.pkl")
    test_df.to_pickle("data/test_data.pkl")
    train_df.to_csv("data/train_data.csv", index=False)
    valid_df.to_csv("data/valid_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")
print(f"Testing samples: {len(test_df)}")

# Check Class Distribution
sns.countplot(data=df, y='labels')
plt.title("Class Distribution")
plt.show()

# Compute Class Weights
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_df['labels']), y=train_df['labels']
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Data Generators
img_size = (224, 224)
batch_size = 32

# Training data generator with augmentation
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8, 1.2]
)

# Validation and testing data generator without augmentation
valid_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)

# Prepare Data Generators
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

# Debugging Outputs
data_batch, label_batch = next(iter(train_generator))
print("Data batch shape:", data_batch.shape)  # Expected: (batch_size, 224, 224, 3)
print("Label batch shape:", label_batch.shape)  # Expected: (batch_size, num_classes)

# Visualize Augmented Images
for i in range(5):  # Display 5 examples
    plt.imshow(data_batch[i])
    plt.title(f"Sample {i + 1}")
    plt.axis('off')
    plt.show()

# Save Class Indices for Flask API
class_indices = train_generator.class_indices
with open("data/class_indices.txt", "w") as f:
    f.write(str(class_indices))
print("Class Indices Mapping saved to class_indices.txt")


# Verification Tests

# Test 1: Verify CSV files
print("\nVerifying CSV files...")
train_csv_exists = os.path.exists("data/train_data.csv")
valid_csv_exists = os.path.exists("data/valid_data.csv")
test_csv_exists = os.path.exists("data/test_data.csv")
print(f"Train CSV exists: {train_csv_exists}")
print(f"Validation CSV exists: {valid_csv_exists}")
print(f"Test CSV exists: {test_csv_exists}")

# Test 2: Inspect Class Distribution
print("\nInspecting class distribution for training dataset...")
sns.countplot(data=train_df, y='labels')
plt.title("Training Set Class Distribution")
plt.show()

# Test 3: Visualize Augmentation (Already Included Above)

# Test 4: Verify Data Generator Outputs
print("\nVerifying data generator outputs...")
print(f"Train Generator Classes: {train_generator.class_indices}")
print(f"Sample data batch shape: {data_batch.shape}")
print(f"Sample label batch shape: {label_batch.shape}")

print("\nAll tests completed.")
