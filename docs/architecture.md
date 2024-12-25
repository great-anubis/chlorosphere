CNN Architecture Framework

Input Shape: (224, 224, 3)
Layers:
Conv2D (32 filters, kernel 3x3) → BatchNormalization → MaxPooling (2x2)
Conv2D (64 filters, kernel 3x3) → BatchNormalization → MaxPooling (2x2)
Conv2D (128 filters, kernel 3x3) → BatchNormalization → MaxPooling (2x2)
Flatten → Dense (256 neurons) → Dropout (0.5) → Output (softmax)
Parameters:
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Categorical Crossentropy
Metrics: Accuracy
File Path: model/cnn_model.json.
