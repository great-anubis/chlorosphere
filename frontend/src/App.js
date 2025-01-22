import React, { useState } from "react";
import Upload from "./components/Upload";
import PredictionResult from "./components/PredictionResult";

const App = () => {
  const [filePath, setFilePath] = useState("");

  return (
    <div>
      <h1>Chlorosphere: Plant Disease Detection</h1>
      <Upload onUploadSuccess={(path) => setFilePath(path)} />
      {filePath && <PredictionResult filePath={filePath} />}
    </div>
  );
};

export default App;
