import ast
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2  # Use cv2 for image processing, or Pillow if preferred.

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Hardcoded relative path to the folder where uploaded images are saved
UPLOAD_FOLDER = "../tests"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hardcoded relative path to the trained model
MODEL_PATH = "../model/trained_model.keras"
if not tf.io.gfile.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the model is correctly saved.")

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Hardcoded relative path to class_indices.txt (a Python dictionary)
CLASS_INDICES_PATH = "../data/class_indices.txt"
if not tf.io.gfile.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}. Ensure the file exists.")

# 1. Read the entire file content
# 2. Use ast.literal_eval to safely parse the dictionary
with open(CLASS_INDICES_PATH, 'r') as f:
    file_content = f.read()
    class_indices_dict = ast.literal_eval(file_content)  # Expecting a valid Python dict

# Sort the dictionary by index (the dict value) and extract class names in order
sorted_by_index = sorted(class_indices_dict.items(), key=lambda x: x[1])
CLASS_NAMES = [item[0] for item in sorted_by_index]

print("Loaded Class Names:", CLASS_NAMES)


class UploadImage(Resource):
    def post(self):
        """
        Handles the uploading of an image file.
        Returns the file path if the upload is successful.
        """
        if 'file' not in request.files:
            return {'message': 'No file part in the request'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'message': 'No selected file'}, 400

        # Secure the filename to avoid directory traversal attacks
        filename = secure_filename(file.filename)
        file_path = f"{UPLOAD_FOLDER}/{filename}"

        try:
            file.save(file_path)
        except Exception as e:
            return {'message': f"File saving failed: {str(e)}"}, 500

        return {
            'message': f'File {filename} uploaded successfully',
            'file_path': file_path
        }, 200


class PredictDisease(Resource):
    def post(self):
        """
        Predicts the disease from the uploaded image using the trained CNN model.
        Expects a JSON payload with 'file_path'.
        """
        data = request.get_json()
        if not data or 'file_path' not in data:
            return {'message': 'file_path not provided'}, 400

        file_path = data['file_path']
        if not tf.io.gfile.exists(file_path):
            return {'message': f'File {file_path} does not exist'}, 400

        try:
            # Preprocess the image
            img = cv2.imread(file_path)
            if img is None:
                return {'message': 'Invalid image file or unsupported format.'}, 400

            # Convert BGR to RGB if needed (depends on your model training pipeline)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            img = img / 255.0                  # Normalize pixel values to [0, 1]
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions, axis=1)[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]

            return {
                'predicted_class': predicted_class_name,
                'confidence_score': confidence
            }, 200

        except Exception as e:
            return {'message': f"Prediction failed: {str(e)}"}, 500


# Add endpoints to the Flask-RESTful API
api.add_resource(UploadImage, '/upload')    # Endpoint for uploading images
api.add_resource(PredictDisease, '/predict')  # Endpoint for predicting diseases

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, port=5000)
