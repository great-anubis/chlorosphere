import React, { useState } from "react";
import axios from "axios";

function Upload() {
  const [file, setFile] = useState(null); // Store the uploaded file
  const [message, setMessage] = useState(""); // Store messages (e.g., "File uploaded!")
  const [prediction, setPrediction] = useState(null); // Store prediction results

  // When the user picks a file
  const handleFileChange = (e) => {
    setFile(e.target.files[0]); // Save the file they selected
  };

  // When the user clicks "Upload"
  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file first!"); // If no file, tell the user
      return;
    }

    const formData = new FormData(); // Wrap the file to send it
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setMessage(response.data.message); // Show Flask's message
    } catch (error) {
      setMessage("Error uploading file."); // Show an error if something goes wrong
    }
  };

  // When the user clicks "Predict"
  const handlePredict = async () => {
    if (!file) {
      setMessage("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setPrediction(response.data); // Save Flask's prediction result
    } catch (error) {
      setMessage("Error predicting.");
    }
  };

  return (
    <div>
      <h1>Upload and Predict</h1>
      <input type="file" onChange={handleFileChange} /> {/* Pick a file */}
      <button onClick={handleUpload}>Upload</button> {/* Upload the file */}
      <button onClick={handlePredict}>Predict</button> {/* Get prediction */}
      {message && <p>{message}</p>} {/* Show messages */}
      {prediction && ( // Show prediction results
        <div>
          <h2>Prediction</h2>
          <p>Disease: {prediction.predicted_disease}</p>
          <p>Confidence: {prediction.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default Upload;
