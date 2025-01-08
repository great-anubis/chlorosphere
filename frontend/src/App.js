import React from "react"; // This is the base React import
import Upload from "./components/Upload"; // Import the Upload component we created

function App() {
  return (
    <div>
      <Upload /> {/* Add the Upload component here */}
    </div>
  );
}

export default App; // This lets other files (like index.js) use App
