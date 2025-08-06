import React, { useState } from "react";

export default function ChatBox({ onSubmit }) {
  const [query, setQuery] = useState("");
  const [loadingResponse, setLoadingResponse] = useState(false);
  const [response, setResponse] = useState("");

  // Called when user clicks submit
  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoadingResponse(true);
    setResponse("");

    try {
      const res = await onSubmit(query);
      setResponse(res);
    } catch (err) {
      setResponse(`Error: ${err.message || err}`);
    }

    setLoadingResponse(false);
  };

  return (
    <section>
      <h2>Ask a question about the PDFs</h2>
      <textarea
        rows={4}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type your query here..."
        style={{ width: "100%", padding: 8, fontSize: 16 }}
        disabled={loadingResponse}
      />
      <button
        onClick={handleSubmit}
        disabled={loadingResponse || !query.trim()}
        style={{ marginTop: 8, padding: "8px 16px", fontSize: 16 }}
      >
        {loadingResponse ? "Waiting for response..." : "Submit Query"}
      </button>

      <div
        style={{
          marginTop: 20,
          padding: 12,
          border: "1px solid #ddd",
          borderRadius: 4,
          minHeight: 80,
          whiteSpace: "pre-wrap",
          backgroundColor: "#f9f9f9",
        }}
      >
        {loadingResponse ? "Loading..." : response || "Response will appear here."}
      </div>
    </section>
  );
}