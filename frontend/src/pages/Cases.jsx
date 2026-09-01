import { useEffect, useState } from "react";
import {
  FolderOpen,
  Plus,
  Loader2,
  Trash2,
  AlertCircle,
  CheckCircle,
  Hash,
  Clock,
  FileText,
  MessageSquare,
  X,
} from "lucide-react";
import { API_URL } from "../config";

function Cases() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    loadCases();
  }, []);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const loadCases = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(API_URL + "/cases");
      if (!res.ok) throw new Error("Failed to load cases");
      setCases(await res.json());
    } catch (err) {
      setError(err.message || "Failed to load cases");
    } finally {
      setLoading(false);
    }
  };

  const createCase = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError("");
    setSuccess("");
    try {
      const res = await fetch(API_URL + "/cases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newName.trim(),
          description: newDescription.trim() || null,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Failed to create case");
      }
      const created = await res.json();
      setCases((prev) => [created, ...prev]);
      setSuccess("Case created successfully");
      setNewName("");
      setNewDescription("");
      setShowForm(false);
    } catch (err) {
      setError(err.message || "Failed to create case");
    } finally {
      setCreating(false);
    }
  };

  const deleteCase = async (caseId, caseName) => {
    if (!window.confirm('Delete "' + caseName + '"? This will remove all associated data.')) return;
    setError("");
    setSuccess("");
    try {
      const res = await fetch(API_URL + "/cases/" + caseId, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Failed to delete case");
      }
      setCases((prev) => prev.filter((c) => c.id !== caseId));
      setSuccess("Case deleted successfully");
    } catch (err) {
      setError(err.message || "Failed to delete case");
    }
  };

  const formatDate = (d) => {
    if (!d) return "";
    return new Date(d).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="cases-layout">
      <div className="cases-header">
        <div>
          <h1>Cases</h1>
          <p>Manage your document/RAG cases</p>
        </div>
        <button className="cases-new-btn" onClick={() => setShowForm(!showForm)}>
          {showForm ? <X size={16} /> : <Plus size={16} />}
          <span>{showForm ? "Cancel" : "New Case"}</span>
        </button>
      </div>

      {error && (
        <div className="cases-error">
          <AlertCircle size={15} />
          <span>{error}</span>
          <button onClick={() => setError("")}>dismiss</button>
        </div>
      )}
      {success && (
        <div className="cases-success">
          <CheckCircle size={15} />
          <span>{success}</span>
          <button onClick={() => setSuccess("")}>dismiss</button>
        </div>
      )}

      {showForm && (
        <div className="cases-form-wrapper">
          <form className="cases-form" onSubmit={createCase}>
            <div className="cases-form-group">
              <label>Case Name</label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. Teaching Materials"
                autoFocus
              />
            </div>
            <div className="cases-form-group">
              <label>Description (optional)</label>
              <input
                type="text"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Short description of this case"
              />
            </div>
            <button type="submit" className="cases-form-submit" disabled={creating || !newName.trim()}>
              {creating ? (
                <><Loader2 size={15} className="spin" /><span>Creating...</span></>
              ) : (
                <><Plus size={15} /><span>Create Case</span></>
              )}
            </button>
          </form>
        </div>
      )}

      <div className="cases-content">
        {loading ? (
          <div className="cases-empty">
            <Loader2 size={24} className="spin" />
            <span>Loading cases...</span>
          </div>
        ) : cases.length === 0 ? (
          <div className="cases-empty">
            <div className="cases-empty-icon">
              <FolderOpen size={28} />
            </div>
            <h2>No cases yet</h2>
            <p>Create a case to start uploading documents and asking questions.</p>
            <button className="cases-empty-btn" onClick={() => setShowForm(true)}>
              <Plus size={15} />
              <span>Create Case</span>
            </button>
          </div>
        ) : (
          <div className="cases-grid">
            {cases.map((c) => (
              <div key={c.id} className="cases-card">
                <div className="cases-card-header">
                  <div className="cases-card-icon">
                    <FolderOpen size={18} />
                  </div>
                  <div className="cases-card-info">
                    <h3 className="cases-card-name">{c.name}</h3>
                    <span className="cases-card-id"><Hash size={11} /> ID: {c.id}</span>
                  </div>
                  <button className="cases-card-delete" onClick={(e) => { e.stopPropagation(); deleteCase(c.id, c.name); }} title="Delete case">
                    <Trash2 size={13} />
                  </button>
                </div>
                {c.description && <p className="cases-card-desc">{c.description}</p>}
                <div className="cases-card-footer">
                  <span className="cases-card-date"><Clock size={12} /> Created {formatDate(c.created_at)}</span>
                  <div className="cases-card-actions">
                    <a href="/documents" className="cases-card-action" title="Documents"><FileText size={13} /><span>Documents</span></a>
                    <a href="/chats" className="cases-card-action" title="Chat"><MessageSquare size={13} /><span>Chat</span></a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Cases;