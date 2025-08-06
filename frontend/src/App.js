import React, { useState } from "react";
import FileUpload from "./components/fileUpload";
import ChatBox from "./components/chatBox";

export default function App() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  // Upload files handler
  async function uploadFiles() {
    if (files.length === 0) {
      alert("Please select files first.");
      return;
    }
    setUploading(true);
    setUploadMessage("");

    const formData = new FormData();
    files.forEach(file => formData.append("files", file));

    try {
      const res = await fetch("http://localhost:8000/upload-pdfs", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Upload failed");
      }

      const data = await res.json();
      setUploadMessage(`Uploaded ${data.processed_pdfs.length} files successfully.`);
      setFiles([]);
    } catch (err) {
      setUploadMessage(`Error: ${err.message}`);
    }
    setUploading(false);
  }

  // Query submit handler passed to ChatBox
  async function handleQuerySubmit(query) {
    const res = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Query failed");
    }

    const data = await res.json();
    return data.response;
  }

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", fontFamily: "Arial, sans-serif" }}>
      <h1>PDF Upload & Query Tool</h1>

      <FileUpload files={files} setFiles={setFiles} />

      {files.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <strong>Files to upload:</strong>
          <ul>
            {files.map((file, idx) => (
              <li key={idx}>{file.name}</li>
            ))}
          </ul>
          <button onClick={uploadFiles} disabled={uploading}>
            {uploading ? "Uploading..." : "Upload Files"}
          </button>
          {uploadMessage && <p>{uploadMessage}</p>}
        </div>
      )}

      <hr />

      <ChatBox onSubmit={handleQuerySubmit} />
    </div>
  );
}