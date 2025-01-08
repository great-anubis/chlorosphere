import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths
data_dir = "data/"
train_csv_path = os.path.join(data_dir, "train_data.csv")
valid_csv_path = os.path.join(data_dir, "valid_data.csv")
test_csv_path = os.path.join(data_dir, "test_data.csv")
class_indices_path = os.path.join(data_dir, "class_indices.txt")

# Test 1: Verify CSV File Existence
print("\n--- Test 1: Verify CSV Files ---")
print(f"Train CSV exists: {os.path.exists(train_csv_path)}")
print(f"Validation CSV exists: {os.path.exists(valid_csv_path)}")
print(f"Test CSV exists: {os.path.exists(test_csv_path)}")

# Test 2: Inspect CSV Files
print("\n--- Test 2: Inspect CSV Content ---")
train_df = pd.read_csv(train_csv_path)
valid_df = pd.read_csv(valid_csv_path)
test_df = pd.read_csv(test_csv_path)

print("Training Data (First 5 rows):")
print(train_df.head())

print("\nValidation Data (First 5 rows):")
print(valid_df.head())

print("\nTesting Data (First 5 rows):")
print(test_df.head())

# Test 3: Verify Data Generators
print("\n--- Test 3: Verify Data Generators ---")
img_size = (224, 224)
batch_size = 32

# Create generators
train_gen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size,
    class_mode='categorical', batch_size=batch_size, shuffle=True
)

data_batch, label_batch = next(iter(train_generator))
print("Data batch shape:", data_batch.shape)  # Expected: (batch_size, 224, 224, 3)
print("Label batch shape:", label_batch.shape)  # Expected: (batch_size, num_classes)

# Visualize augmented images
print("\nVisualizing augmented images...")
for i in range(5):  # Display 5 examples
    plt.imshow(data_batch[i])
    plt.title(f"Sample {i + 1}")
    plt.axis('off')
    plt.savefig(f"{data_dir}test_sample_{i + 1}.png")  # Save each sample
    plt.close()

print(f"Augmented sample images saved to {data_dir} as test_sample_1.png, test_sample_2.png, etc.")

# Test 4: Verify Class Indices
print("\n--- Test 4: Verify Class Indices ---")
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = eval(f.read())
        print("Class Indices Mapping:")
        print(class_indices)
else:
    print("Class indices file not found.")
