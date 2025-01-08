import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

# Paths
root_dir = os.getcwd()
model_dir = os.path.join(root_dir, "model/")
data_dir = os.path.join(root_dir, "data/")
docs_dir = os.path.join(root_dir, "docs/")

trained_model_path = os.path.join(model_dir, "trained_model.keras")
metrics_path = os.path.join(docs_dir, "metrics.md")
history_plot_path = os.path.join(docs_dir, "training_history.png")
cnn_model_path = os.path.join(model_dir, "cnn_model.json")
cnn_weights_path = os.path.join(model_dir, "cnn_weights.weights.h5")  # Optional if you plan to load weights

# Ensure Directories Exist
for directory in [model_dir, docs_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Data Generators
train_gen = ImageDataGenerator(rescale=1.0 / 255)
valid_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load Dataset
train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
valid_df = pd.read_csv(os.path.join(data_dir, "valid_data.csv"))

# Data Generator Configurations
img_size = (224, 224)
batch_size = 32

train_generator = train_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels',
    target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=True
)

valid_generator = valid_gen.flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels',
    target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=False
)

# Load CNN Architecture (Sequential)
with open(cnn_model_path, "r") as json_file:
    model = model_from_json(json_file.read())
print("Loaded CNN architecture from cnn_model.json.")

# Build the model to set input shape
# (Alternatively, you can do a forward pass with a dummy tensor)
model.build(input_shape=(None, 224, 224, 3))

# (Optional) If you have pretrained weights you want to load:
# model.load_weights(cnn_weights_path)

# Remove the last layer
model.pop()

# Add a new classification layer for 38 classes with a unique name
model.add(Dense(38, activation='softmax', name="final_output"))


# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    trained_model_path, monitor='val_accuracy',
    save_best_only=True, verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train Model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save Final Model
model.save(trained_model_path)
print(f"Trained model saved to {trained_model_path}")

# Plot Training History
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(history_plot_path)
plt.close()
print(f"Training history plot saved to {history_plot_path}")

# Save Metrics
final_training_accuracy = history.history['accuracy'][-1]
final_validation_accuracy = history.history['val_accuracy'][-1]
with open(metrics_path, "w") as f:
    f.write("# Model Metrics\n")
    f.write(f"Final Training Accuracy: {final_training_accuracy:.4f}\n")
    f.write(f"Final Validation Accuracy: {final_validation_accuracy:.4f}\n")
print(f"Metrics saved to {metrics_path}")
