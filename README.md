Chlorosphere: Plant Disease Detection System

Overview

Chlorosphere is a machine learning-powered system designed to detect plant diseases from leaf images. By leveraging a custom-built Convolutional Neural Network (CNN) and a user-friendly interface, Chlorosphere aims to assist farmers, researchers, and hobbyists in identifying plant health issues efficiently and accurately.

This system operates locally and provides an integrated solution for image analysis and disease prediction, with a focus on scalability and customization for future enhancements.

Features

Custom CNN Model: Built from scratch to classify plant diseases with high accuracy.
Image Upload and Analysis: Upload plant leaf images to receive disease predictions and confidence scores.
Preprocessed Dataset: Efficiently handles large datasets with preprocessing steps like resizing, normalization, and augmentation.
Local Integration: Combines a Flask backend with a React frontend for seamless operation on local machines.

Tech Stack

Programming Language: Python, JavaScript
Machine Learning: TensorFlow/Keras (custom CNN model)
Backend: Flask (API endpoints for image uploads and predictions)
Frontend: React (for interactive UI and dynamic result display)
Data Processing: OpenCV, pandas, NumPy
Testing: pytest, unittest

Project Structure

Plant-Disease-Detection/
│
├── data/
│   ├── raw/            # Raw dataset
│   ├── processed/      # Preprocessed dataset
│
├── notebooks/
│   ├── eda.ipynb       # Exploratory data analysis
│   ├── model.ipynb     # Custom CNN training and evaluation
│
├── model/
│   ├── trained_model.h5  # Saved trained model
│   ├── utils.py          # Helper functions
│
├── backend/
│   ├── app.py           # Flask application
│   ├── requirements.txt # Python dependencies
│
├── frontend/
│   ├── public/          # Static assets
│   ├── src/             # React components
│   │   ├── App.js       # Main React app
│   │   ├── Upload.js    # File upload component
│   │   ├── Results.js   # Results display component
│   ├── package.json     # Node.js dependencies
│
├── docs/
│   ├── architecture.md  # System architecture documentation
│   ├── metrics.md       # Model performance metrics
│
├── README.md            # Project overview

How It Works

Dataset Preparation: The PlantVillage dataset is preprocessed using resizing, normalization, and augmentation techniques.
Model Training: A custom CNN is designed and trained on the dataset to classify various plant diseases.
API Backend: Flask APIs handle image uploads and forward them to the trained model for predictions.
Frontend Interface: Users upload images via a React-based web interface and receive disease predictions along with confidence scores.

Future Enhancements
Expand the model to support additional plant species and diseases.
Add a feature to suggest remedies or solutions based on predictions.
Optimize the system for deployment on mobile devices or edge computing platforms.

Contributors
Great: Machine Learning Developer
Deborah: Frontend/Backend Developer
Carlentz: Data Engineer

License
This project is licensed under the MIT License. See the LICENSE file for details.
