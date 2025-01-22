import ast
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS  # Added for Cross-Origin Resource Sharing
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2  # For image preprocessing.

# Initialize Flask app and RESTful API
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"])

api = Api(app)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# Relative path for storing uploaded images
UPLOAD_FOLDER = "../backend/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Relative path for the trained CNN model
MODEL_PATH = "../model/trained_model.keras"
if not tf.io.gfile.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the model is correctly saved.")

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Relative path for the class indices file
CLASS_INDICES_PATH = "../data/class_indices.txt"
if not tf.io.gfile.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}. Ensure the file exists.")

# Load and parse class indices into a Python dictionary
with open(CLASS_INDICES_PATH, 'r') as f:
    file_content = f.read()
    class_indices_dict = ast.literal_eval(file_content)  # Safely parse the file content

# Sort the class indices and extract class names
sorted_by_index = sorted(class_indices_dict.items(), key=lambda x: x[1])
CLASS_NAMES = [item[0] for item in sorted_by_index]

print("Loaded Class Names:", CLASS_NAMES)


class UploadImage(Resource):
    def post(self):
        """
        Handles the uploading of an image file.
        Saves the file to the UPLOAD_FOLDER and returns its file path.
        """
        if 'file' not in request.files:
            print("No file part in the request.")
            return {'message': 'No file part in the request'}, 400

        file = request.files['file']
        if file.filename == '':
            print("No selected file.")
            return {'message': 'No selected file'}, 400

        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        file_path = f"{UPLOAD_FOLDER}/{filename}"
        print(f"Saving file to: {file_path}")

        try:
            file.save(file_path)
        except Exception as e:
            print(f"File saving failed: {str(e)}")
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
            print("No file_path provided.")
            return {'message': 'file_path not provided'}, 400

        file_path = data['file_path']
        print(f"Received file_path: {file_path}")

        if not tf.io.gfile.exists(file_path):
            print(f"File does not exist: {file_path}")
            return {'message': f'File {file_path} does not exist'}, 400

        try:
            # Preprocess the image
            img = cv2.imread(file_path)
            if img is None:
                print("Invalid image file or unsupported format.")
                return {'message': 'Invalid image file or unsupported format.'}, 400

            # Convert BGR to RGB (depends on model training pipeline)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Resize to match input size of the CNN
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
            print(f"Prediction failed: {str(e)}")
            return {'message': f"Prediction failed: {str(e)}"}, 500


# Add endpoints to the Flask-RESTful API
api.add_resource(UploadImage, '/upload')    # Endpoint for uploading images
api.add_resource(PredictDisease, '/predict')  # Endpoint for predicting diseases

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, port=5000)
