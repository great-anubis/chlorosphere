import React, { useState } from "react";
import axios from "axios";

const Upload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]); // Set the selected file
  };

  const handleUpload = async () => {
    if (!file) {
        setMessage("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });
        console.log("Upload Response:", response.data); // Debugging log
        setMessage(response.data.message); // Display upload message
        onUploadSuccess(response.data.file_path); // Pass file_path to parent
    } catch (error) {
        console.error("Upload Error:", error.response || error);
        setMessage(error.response?.data?.message || "File upload failed.");
    }
};


  return (
    <div>
      <h2>Upload Plant Image</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {message && <p>{message}</p>}
    </div>
  );
};

export default Upload;
