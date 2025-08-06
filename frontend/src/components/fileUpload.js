import React, { useRef } from "react";

const MAX_FILES = 10;

export default function FileUpload({ files, setFiles }) {
  const fileInputRef = useRef(null);

  const onDrop = (e) => {
    e.preventDefault();
    let droppedFiles = Array.from(e.dataTransfer.files).filter(f =>
      f.name.toLowerCase().endsWith(".pdf")
    );

    if (files.length + droppedFiles.length > MAX_FILES) {
      alert(`Max ${MAX_FILES} files allowed total.`);
      return;
    }

    setFiles(prev => [...prev, ...droppedFiles]);
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  const onFileSelect = (e) => {
    let selectedFiles = Array.from(e.target.files).filter(f =>
      f.name.toLowerCase().endsWith(".pdf")
    );

    if (files.length + selectedFiles.length > MAX_FILES) {
      alert(`Max ${MAX_FILES} files allowed total.`);
      return;
    }

    setFiles(prev => [...prev, ...selectedFiles]);
  };

  return (
    <>
      <section
        onDrop={onDrop}
        onDragOver={onDragOver}
        style={{
          border: "2px dashed #666",
          padding: 20,
          borderRadius: 8,
          textAlign: "center",
          color: "#666",
          marginBottom: 10,
          cursor: "pointer",
        }}
        onClick={() => fileInputRef.current.click()}
      >
        Drag & drop PDF files here, or click to select
      </section>

      <input
        type="file"
        multiple
        accept=".pdf"
        onChange={onFileSelect}
        ref={fileInputRef}
        style={{ display: "none" }}
      />
    </>
  );
}