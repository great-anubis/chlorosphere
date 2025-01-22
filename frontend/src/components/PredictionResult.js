import React, { useState } from "react";
import axios from "axios";

const PredictionResult = ({ filePath }) => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPrediction = async () => {
    setLoading(true);
    setError("");
    try {
        const response = await axios.post("http://127.0.0.1:5000/predict", {
            file_path: filePath,
        });
        console.log("Prediction Response:", response.data); // Debugging log
        setResult(response.data); // Store prediction result
    } catch (err) {
        console.error("Prediction Error:", err.response || err);
        setError(err.response?.data?.message || "Prediction failed.");
    } finally {
        setLoading(false);
    }
};


  return (
    <div>
      <h2>Prediction Result</h2>
      {filePath && <button onClick={fetchPrediction}>Get Prediction</button>}
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <div>
          <p><strong>Predicted Class:</strong> {result.predicted_class}</p>
          <p><strong>Confidence Score:</strong> {result.confidence_score.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
