# Chlorosphere: Plant Disease Detection System

## Objective
Develop a machine learning-powered system to detect plant diseases from leaf images. This includes creating a Convolutional Neural Network (CNN) from scratch, building a robust data pipeline, and integrating it into a user-friendly web application for local use.

---

## Tech Stack

### **Data Processing**
- **Programming Language:** Python
- **Libraries:**
  - **NumPy:** For numerical computations.
  - **pandas:** For dataset management.
  - **OpenCV:** For image preprocessing and transformations.
  - **Matplotlib:** For visualizing data distribution and results.

### **Machine Learning**
- **Programming Language:** Python
- **Framework:** TensorFlow/Keras  
  TensorFlow is selected for its robustness, community support, and extensive features for model building and training.
- **Features:**
  - Custom CNN architecture with convolutional layers, pooling, and dense layers.
  - Training with Adam optimizer and early stopping to avoid overfitting.

### **Backend**
- **Framework:** Flask  
  Flask is selected for its simplicity and ease of integration with Python-based ML models.
- **Libraries:**
  - **Flask-RESTful:** For creating API endpoints.
  - **Werkzeug:** For handling file uploads.
  - **joblib:** For saving and loading the trained CNN model.

### **Frontend**
- **Framework:** React  
  React is chosen for its flexibility, component-based architecture, and efficient rendering.
- **Technologies:**
  - **React Hooks:** For managing state and lifecycle in functional components.
  - **Axios:** For making HTTP requests to the Flask backend.
  - **Material-UI (optional):** For polished, pre-styled components.

### **Testing**
- **Frameworks:**
  - **pytest:** For unit testing of backend APIs.
  - **unittest:** For testing model predictions.
- **Tools:**
  - Sample plant leaf images for functional testing.

---
## Structure
- Chlorosphere/

    - backend/
    - data/
        - raw/             # Raw dataset
        - processed/       # Preprocessed dataset
    - docs/
        - architecture.md  # System architecture documentation
        - metrics.md       # Model performance metrics
    - frontend/
        - public/          # Static assets
        - src/             # React components
            -    components/  # Components
            -    Styles/      # Frontend styles
            -    App.js       # Main React app
        - package.json     # Node.js dependencies
    - model/
    - notebooks/
    - README.md            # Project overview
---

## Key Notes
- **Frontend and Backend Integration:** Use Axios in React to make API calls to Flask endpoints for smooth communication.
- **No Deployment:** The system is designed for local use and testing.
- **Focus on Custom Model Development:** Emphasis on building a CNN from scratch to solidify ML understanding.

## Team
- Great
- Deborah
- Carlentz
- Nutifafa