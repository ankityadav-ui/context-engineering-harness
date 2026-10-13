// Import React state management.
import { useState } from "react";

// Import the stylesheet for the application.
import "./App.css";
import MaskingPanel from "./MaskingPanel";

function App() {
  // Store the current chat input.
  const [message, setMessage] = useState("");

  // Store all messages displayed in the chat.
  const [messages, setMessages] = useState([]);

  // Track whether a chat request is currently running.
  const [loading, setLoading] = useState(false);

  // Store the currently selected PDF file.
  const [selectedFile, setSelectedFile] = useState(null);

  // Store the ID returned after a document is uploaded.
  const [documentId, setDocumentId] = useState("");

  // Track whether document upload is currently running.
  const [uploading, setUploading] = useState(false);

  // Store document-upload errors.
  const [uploadError, setUploadError] = useState("");

  // Send a message to the FastAPI backend.
  const sendMessage = async () => {
    // Do not send an empty message or another request while loading.
    if (!message.trim() || loading) {
      return;
    }

    // Remove unnecessary whitespace from the user's message.
    const userMessage = message.trim();

    // Immediately display the user's message in the chat.
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    // Clear the input box.
    setMessage("");

    // Show the loading state.
    setLoading(true);

    try {
      // Send the message to the FastAPI chat endpoint.
      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: userMessage,
          }),
        }
      );

      // Convert the backend response into JSON.
      const data = await response.json();

      // Handle HTTP errors returned by the backend.
      if (!response.ok) {
        throw new Error(
          data.detail || "Something went wrong."
        );
      }

      // Add the assistant's response to the chat.
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: data.response,
        },
      ]);
    } catch (error) {
      // Display a readable error message inside the chat.
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content:
            error.message ||
            "Unable to connect to the backend.",
        },
      ]);
    } finally {
      // Stop the loading state regardless of success or failure.
      setLoading(false);
    }
  };


  // Allow Enter to send a message.
  // Shift + Enter is still available for a new line.
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };


  // Handle selection of a PDF file.
  const handleFileChange = (event) => {
    // Get the first selected file.
    const file = event.target.files[0];

    // Clear previous upload messages.
    setUploadError("");

    // Clear the previous document ID.
    setDocumentId("");

    // Handle the case where no file was selected.
    if (!file) {
      setSelectedFile(null);
      return;
    }

    // Check whether the selected file is a PDF.
    const isPdf =
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf");

    // Reject non-PDF files.
    if (!isPdf) {
      setSelectedFile(null);
      setUploadError("Please select a PDF file.");
      return;
    }

    // Store the selected PDF.
    setSelectedFile(file);
  };


  // Upload the selected PDF to the FastAPI backend.
  const handleUpload = async () => {
    // Do not start an upload without a selected file.
    if (!selectedFile) {
      setUploadError("Please select a PDF first.");
      return;
    }

    // Start the uploading state.
    setUploading(true);

    // Clear previous upload errors and document ID.
    setUploadError("");
    setDocumentId("");

    try {
      // FormData is used because the backend expects a file upload.
      const formData = new FormData();

      // Add the selected PDF using the field name expected by FastAPI.
      formData.append("file", selectedFile);

      // Send the PDF to the document upload endpoint.
      const response = await fetch(
        "http://127.0.0.1:8000/documents/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      // Convert the backend response into JSON.
      const data = await response.json();

      // Handle upload errors returned by the backend.
      if (!response.ok) {
        throw new Error(
          data.detail || "Document upload failed."
        );
      }

      // Store the generated document ID.
      setDocumentId(data.document_id);
    } catch (error) {
      // Display a readable upload error.
      setUploadError(
        error.message ||
        "Unable to upload the document."
      );
    } finally {
      // Stop the uploading state.
      setUploading(false);
    }
  };


  return (
    <div className="app">

      {/* Main application heading. */}
      <h1>Context Engineering Harness</h1>


      {/* Document upload section. */}
      <section className="upload-section">

        {/* Section heading. */}
        <h2>Upload Document</h2>

        {/* PDF file selector. */}
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileChange}
        />

        {/* Display the selected filename. */}
        {selectedFile && (
          <p>
            Selected:{" "}
            <strong>{selectedFile.name}</strong>
          </p>
        )}

        {/* Upload the selected PDF. */}
        <button
          onClick={handleUpload}
          disabled={!selectedFile || uploading}
        >
          {uploading
            ? "Processing..."
            : "Upload PDF"}
        </button>

        {/* Display document-upload errors. */}
        {uploadError && (
          <p className="error">
            {uploadError}
          </p>
        )}

        {/* Display the generated document ID. */}
        {documentId && (
          <div>
            <p>
              Document ID:{" "}
              <strong>{documentId}</strong>
            </p>

            {/* Explain what has happened to the document. */}
            <p>
              Document uploaded and processed successfully.
            </p>
          </div>
        )}

      </section>


      {/* Chat section. */}
      <section className="chat-section">

        {/* Section heading. */}
        <h2>Chat</h2>


        {/* Container containing the conversation. */}
        <div className="chat-box">

          {/* Render every message in the conversation. */}
          {messages.map((item, index) => (
            <div
              key={index}
              className={`message ${item.role}`}
            >
              {/* Display the speaker name. */}
              <strong>
                {item.role === "user"
                  ? "You"
                  : "Assistant"}
                :
              </strong>

              {/* Display the actual message. */}
              <p>{item.content}</p>
            </div>
          ))}


          {/* Display a temporary loading message while the LLM responds. */}
          {loading && (
            <div className="message assistant">
              <strong>Assistant:</strong>
              <p>Thinking...</p>
            </div>
          )}

        </div>


        {/* Chat input area. */}
        <div className="chat-input">

          {/* Text box used to enter the user's question. */}
          <textarea
            value={message}
            onChange={(event) =>
              setMessage(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask something..."
            disabled={loading}
            rows={3}
          />

          {/* Send the current message. */}
          <button
            onClick={sendMessage}
            disabled={
              loading || !message.trim()
            }
          >
            {loading
              ? "Sending..."
              : "Send"}
          </button>

        </div>

      </section>

    </div>
  );
}


// Export the application component.
export default App;


