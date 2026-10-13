
import { useState } from "react";
import "./MaskingPanel.css";

const API_URL = "http://127.0.0.1:8000/masking/text";

function MaskingPanel() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleMask = async () => {
    // Reset previous results and validate the input.
    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Please enter some text to mask.");
      return;
    }

    setLoading(true);

    try {
      // Send the text to the FastAPI masking endpoint.
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      // Read the API response.
      const data = await response.json();

      // Display errors returned by the backend.
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to mask the text."
        );
      }

      // Store only the response fields needed by the UI.
      setResult({
        maskedText: data.masked_text,
        safetyPassed: data.safety_passed,
        detectedCount: data.detected_count,
        categories: data.detected_categories || [],
        safetyReasons: data.safety_reasons || [],
      });
    } catch (err) {
      // Handle network errors and backend errors.
      setError(
        err.message || "Cannot connect to the masking server."
      );
    } finally {
      // Re-enable the button after the request finishes.
      setLoading(false);
    }
  };

  return (
    <div className="masking-panel">
      <h2>Sensitive Document Masking</h2>

      <p>
        Enter text containing sensitive information. The system
        will detect supported entities, replace them with
        placeholders, and run a secondary safety check.
      </p>

      <label htmlFor="masking-input">
        Text to mask
      </label>

      <textarea
        id="masking-input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Example: Rahul Sharma can be contacted at rahul@example.com or 9876543210."
        rows={7}
        maxLength={50000}
      />

      <div className="masking-actions">
        <button onClick={handleMask} disabled={loading}>
          {loading ? "Masking..." : "Mask Sensitive Information"}
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={() => {
            setText("");
            setResult(null);
            setError("");
          }}
          disabled={loading}
        >
          Clear
        </button>
      </div>

      {error && (
        <div className="masking-error" role="alert">
          {error}
        </div>
      )}

      {result && (
        <section className="masking-result">
          <h3>Masked Output</h3>

          <pre>{result.maskedText}</pre>

          <h3>Detection Summary</h3>

          <p>
            <strong>Entities detected:</strong>{" "}
            {result.detectedCount}
          </p>

          <p>
            <strong>Categories:</strong>{" "}
            {result.categories.length > 0
              ? result.categories.join(", ")
              : "None"}
          </p>

          <h3>Safety Check</h3>

          <p
            className={
              result.safetyPassed
                ? "safety-passed"
                : "safety-failed"
            }
            role="status"
          >
            {result.safetyPassed
              ? "Passed — configured sensitive patterns were not detected."
              : "Failed — possible sensitive information remains."}
          </p>

          {result.safetyReasons.length > 0 && (
            <ul className="masking-error">
              {result.safetyReasons.map((reason, index) => (
                <li key={`${reason}-${index}`}>{reason}</li>
              ))}
            </ul>
          )}

          <p className="masking-note">
            A passed check does not guarantee complete
            anonymization. Review the output before using it.
          </p>
        </section>
      )}
    </div>
  );
}

export default MaskingPanel;
