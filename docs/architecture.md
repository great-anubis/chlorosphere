### CNN Architecture Framework

**Input Shape:**
- (224, 224, 3)

**Layers:**
1. **Convolutional Block 1:**
   - Conv2D (16 filters, kernel size 3x3, activation='relu', padding='same')
   - BatchNormalization
   - MaxPooling2D (pool size 2x2)

2. **Convolutional Block 2:**
   - Conv2D (32 filters, kernel size 3x3, activation='relu', padding='same')
   - BatchNormalization
   - MaxPooling2D (pool size 2x2)

3. **Convolutional Block 3:**
   - Conv2D (64 filters, kernel size 3x3, activation='relu', padding='same')
   - BatchNormalization
   - MaxPooling2D (pool size 2x2)

4. **Convolutional Block 4:**
   - Conv2D (128 filters, kernel size 3x3, activation='relu', padding='same')
   - BatchNormalization
   - MaxPooling2D (pool size 2x2)

5. **Convolutional Block 5:**
   - Conv2D (256 filters, kernel size 3x3, activation='relu', padding='same')
   - BatchNormalization
   - MaxPooling2D (pool size 2x2)

6. **Fully Connected Layers:**
   - Flatten
   - Dense (256 neurons, activation='relu')
   - Dropout (rate=0.5)
   - Dense (number of classes, activation='softmax')

**Parameters:**
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

**File Path:**
- Model saved as: `model/cnn_model.json`

**Key Notes:**
- This architecture balances simplicity and scalability, supporting a dataset with 20.6k images across 15 classes.
- BatchNormalization stabilizes and accelerates training by normalizing activations.
- Dropout prevents overfitting by randomly deactivating neurons during training.
- Softmax ensures the output represents class probabilities.
