import { useEffect, useRef, useState } from "react";
import {
  FileText,
  Upload,
  Loader2,
  Trash2,
  ChevronDown,
  ChevronRight,
  Settings2,
  Clock,
  Hash,
  AlertCircle,
  CheckCircle,
  File,
} from "lucide-react";
import { casesApi } from "../api/cases";
import { documentsApi } from "../api/documents";
import { ApiError } from "../api/client";
const SUPPORTED_TYPES = [".pdf", ".txt", ".docx"];

function Documents() {
  const [caseId, setCaseId] = useState("");
  const [cases, setCases] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docDetails, setDocDetails] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [expandedChunks, setExpandedChunks] = useState({});
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => { loadCases(); }, []);

  useEffect(() => {
    if (caseId) {
      loadDocuments();
      setSelectedDoc(null);
      setDocDetails(null);
      setChunks([]);
      setExpandedChunks({});
    }
  }, [caseId]);

  const loadCases = async () => {
    try {
      const data = await casesApi.list();
      setCases(data);
      if (data.length > 0 && !caseId) {
        setCaseId(String(data[0].id));
      }
    } catch (err) { console.error("Failed to load cases:", err); }
  };

  const loadDocuments = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await documentsApi.list(caseId);
      setDocuments(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to load documents");
      }
    }
    finally { setLoading(false); }
  };

  const loadDocumentDetails = async (docId) => {
    setSelectedDoc(docId);
    setDocDetails(null);
    setChunks([]);
    setExpandedChunks({});
    try {
      const data = await documentsApi.getDetails(caseId, docId);
      setDocDetails(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to load document details");
      }
    }
    try {
      const data = await documentsApi.getChunks(caseId, docId);
      setChunks(data.chunks || []);
    } catch (err) { console.error("Failed to load chunks:", err); }
  };

  const uploadFile = async (file) => {
    if (!file) return;
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!SUPPORTED_TYPES.includes(ext)) {
      setError("Unsupported file type: " + ext + ". Supported: " + SUPPORTED_TYPES.join(", "));
      return;
    }
    setUploading(true);
    setError("");
    setSuccess("");
    try {
      const data = await documentsApi.upload(caseId, file);
      setSuccess("Uploaded " + file.name + " (" + data.chunk_count + " chunks)");
      loadDocuments();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Upload failed");
      }
    }
    finally { setUploading(false); }
  };

  const deleteDocument = async (docId, filename) => {
    if (!window.confirm('Delete "' + filename + '"? This cannot be undone.')) return;
    setError("");
    setSuccess("");
    try {
      await documentsApi.delete(caseId, docId);
      setSuccess("Deleted " + filename);
      if (selectedDoc === docId) { setSelectedDoc(null); setDocDetails(null); setChunks([]); }
      loadDocuments();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Delete failed");
      }
    }
  };

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setDragOver(false); };
  const handleDrop = (e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); };
  const handleFileSelect = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; };
  const toggleChunk = (id) => setExpandedChunks((p) => ({ ...p, [id]: !p[id] }));
  const getFileType = (fn) => fn.split(".").pop().toLowerCase();
  const formatDate = (d) => { if (!d) return ""; return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" }); };

  return (
    <div className="docs-layout">
      <div className="docs-sidebar">
        <div className="docs-sidebar-header"><h3>Documents</h3></div>
        <div className="docs-case-selector">
          <Settings2 size={14} />
          <span>Case</span>
          <select value={caseId} onChange={(e) => setCaseId(e.target.value)}>
            {cases.map((c) => (<option key={c.id} value={c.id}>{c.name}</option>))}
          </select>
        </div>
        <div className={"docs-upload-area" + (dragOver ? " drag-over" : "")}
          onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}>
          <input ref={fileInputRef} type="file" accept=".pdf,.txt,.docx" onChange={handleFileSelect} style={{ display: "none" }} />
          {uploading ? (
            <><Loader2 size={20} className="spin" /><span>Processing document...</span></>
          ) : (
            <><Upload size={20} /><span>Drop a file here or click to browse</span><span className="docs-upload-hint">PDF, TXT, DOCX</span></>
          )}
        </div>
        <div className="docs-list">
          {loading ? (
            <div className="docs-list-empty"><Loader2 size={16} className="spin" /><span>Loading...</span></div>
          ) : documents.length === 0 ? (
            <div className="docs-list-empty"><FileText size={16} /><span>No documents yet</span></div>
          ) : documents.map((doc) => (
            <div key={doc.id} className={"docs-list-item" + (selectedDoc === doc.id ? " active" : "")}
              onClick={() => loadDocumentDetails(doc.id)}>
              <div className="docs-list-item-icon">
                {getFileType(doc.filename) === "pdf" ? <FileText size={16} /> : <File size={16} />}
              </div>
              <div className="docs-list-item-info">
                <span className="docs-list-item-name">{doc.filename}</span>
                <span className="docs-list-item-meta">ID: {doc.id} &middot; {formatDate(doc.created_at)}</span>
              </div>
              <button className="docs-list-item-delete" onClick={(e) => { e.stopPropagation(); deleteDocument(doc.id, doc.filename); }} title="Delete document">
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="docs-main">
        {error && <div className="docs-error"><AlertCircle size={15} /><span>{error}</span><button onClick={() => setError("")}>dismiss</button></div>}
        {success && <div className="docs-success"><CheckCircle size={15} /><span>{success}</span><button onClick={() => setSuccess("")}>dismiss</button></div>}
        {!selectedDoc ? (
          <div className="docs-empty">
            <div className="docs-empty-icon"><FileText size={28} /></div>
            <h2>Select a document</h2>
            <p>Choose a document from the list to view its details, extracted text, and chunks.</p>
          </div>
        ) : !docDetails ? (
          <div className="docs-empty"><Loader2 size={24} className="spin" /><span>Loading document...</span></div>
        ) : (
          <div className="docs-detail">
            <div className="docs-detail-header">
              <div className="docs-detail-title"><FileText size={20} /><h2>{docDetails.filename}</h2></div>
              <div className="docs-detail-meta">
                <span><Hash size={13} /> ID: {docDetails.id}</span>
                <span><Settings2 size={13} /> Case: {docDetails.case_id}</span>
                <span><Clock size={13} /> {formatDate(docDetails.created_at)}</span>
              </div>
            </div>
            <div className="docs-section">
              <h3>Extracted Text</h3>
              <div className="docs-text-preview">
                {docDetails.text_content || <span className="docs-text-empty">No text extracted</span>}
              </div>
            </div>
            {chunks.length > 0 && (
              <div className="docs-section">
                <h3>Chunks ({chunks.length})</h3>
                <div className="docs-chunks">
                  {chunks.map((chunk) => (
                    <div key={chunk.id} className="docs-chunk">
                      <div className="docs-chunk-header" onClick={() => toggleChunk(chunk.id)}>
                        {expandedChunks[chunk.id] ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                        <span>Chunk {chunk.chunk_index}</span>
                        <span className="docs-chunk-id">ID: {chunk.id}</span>
                      </div>
                      {expandedChunks[chunk.id] && <div className="docs-chunk-content">{chunk.text}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Documents;
