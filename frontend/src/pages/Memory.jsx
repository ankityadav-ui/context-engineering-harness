import { useEffect, useState } from "react";
import {
  Brain,
  Loader2,
  Plus,
  Trash2,
  AlertCircle,
  CheckCircle,
  Tag,
  Clock,
  Star,
  Edit3,
  X,
  Save,
  Info,
} from "lucide-react";
import { memoryApi } from "../api/memory";
import { ApiError } from "../api/client";

const MEMORY_TYPES = [
  { value: "fact", label: "Fact", color: "blue" },
  { value: "preference", label: "Preference", color: "green" },
  { value: "context", label: "Context", color: "purple" },
  { value: "note", label: "Note", color: "orange" },
  { value: "goal", label: "Goal", color: "red" },
];

function Memory() {
  const [memories, setMemories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [newContent, setNewContent] = useState("");
  const [newType, setNewType] = useState("fact");
  const [newImportance, setNewImportance] = useState(0.5);
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editContent, setEditContent] = useState("");
  const [editType, setEditType] = useState("fact");
  const [editImportance, setEditImportance] = useState(0.5);

  useEffect(() => {
    loadMemories();
  }, []);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const loadMemories = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await memoryApi.list();
      setMemories(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to load memories");
      }
    } finally {
      setLoading(false);
    }
  };

  const createMemory = async (e) => {
    e.preventDefault();
    if (!newContent.trim()) return;
    setCreating(true);
    setError("");
    setSuccess("");
    try {
      await memoryApi.create({
        content: newContent.trim(),
        memory_type: newType,
        importance: newImportance,
      });
      setSuccess("Memory created successfully");
      setNewContent("");
      setNewType("fact");
      setNewImportance(0.5);
      setShowForm(false);
      await loadMemories();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to create memory");
      }
    } finally {
      setCreating(false);
    }
  };

  const startEdit = (memory) => {
    setEditingId(memory.id);
    setEditContent(memory.content);
    setEditType(memory.memory_type);
    setEditImportance(memory.importance);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditContent("");
    setEditType("fact");
    setEditImportance(0.5);
  };

  const saveEdit = async (memoryId) => {
    if (!editContent.trim()) return;
    setError("");
    setSuccess("");
    try {
      await memoryApi.update(memoryId, {
        content: editContent.trim(),
        memory_type: editType,
        importance: editImportance,
      });
      setSuccess("Memory updated successfully");
      setEditingId(null);
      await loadMemories();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to update memory");
      }
    }
  };

  const deleteMemory = async (memoryId) => {
    if (!window.confirm("Delete this memory? This cannot be undone.")) return;
    setError("");
    setSuccess("");
    try {
      await memoryApi.delete(memoryId);
      setSuccess("Memory deleted successfully");
      await loadMemories();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to delete memory");
      }
    }
  };

  const formatDate = (d) => {
    if (!d) return "";
    return new Date(d).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getTypeInfo = (type) => {
    return MEMORY_TYPES.find((t) => t.value === type) || MEMORY_TYPES[0];
  };

  return (
    <div className="memory-layout">
      <div className="memory-sidebar">
        <div className="memory-sidebar-header">
          <h3>Long-Term Memory</h3>
        </div>

        <button
          className="memory-new-btn"
          onClick={() => setShowForm(!showForm)}
        >
          {showForm ? <X size={16} /> : <Plus size={16} />}
          <span>{showForm ? "Cancel" : "New Memory"}</span>
        </button>

        {showForm && (
          <div className="memory-form-wrapper">
            <form className="memory-form" onSubmit={createMemory}>
              <div className="memory-form-group">
                <label>Memory Content</label>
                <textarea
                  value={newContent}
                  onChange={(e) => setNewContent(e.target.value)}
                  placeholder="e.g. User prefers concise explanations"
                  rows={3}
                  autoFocus
                />
              </div>
              <div className="memory-form-group">
                <label>Type</label>
                <select
                  value={newType}
                  onChange={(e) => setNewType(e.target.value)}
                >
                  {MEMORY_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="memory-form-group">
                <label>Importance ({newImportance.toFixed(1)})</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={newImportance}
                  onChange={(e) => setNewImportance(parseFloat(e.target.value))}
                />
              </div>
              <button
                type="submit"
                className="memory-form-submit"
                disabled={creating || !newContent.trim()}
              >
                {creating ? (
                  <>
                    <Loader2 size={15} className="spin" />
                    <span>Creating...</span>
                  </>
                ) : (
                  <>
                    <Plus size={15} />
                    <span>Add Memory</span>
                  </>
                )}
              </button>
            </form>
          </div>
        )}

        <div className="memory-sidebar-info">
          <Info size={14} />
          <span>
            These are persistent memories that persist across all chats.
            They help the AI understand your preferences and context.
          </span>
        </div>
      </div>

      <div className="memory-main">
        <div className="memory-header">
          <h1>
            <Brain size={22} />
            Memory Manager
          </h1>
          <p className="memory-subtitle">
            Manage long-term memories that help the AI understand your
            preferences, facts, and context across conversations.
          </p>
        </div>

        {error && (
          <div className="memory-error">
            <AlertCircle size={15} />
            <span>{error}</span>
            <button onClick={() => setError("")}>dismiss</button>
          </div>
        )}
        {success && (
          <div className="memory-success">
            <CheckCircle size={15} />
            <span>{success}</span>
            <button onClick={() => setSuccess("")}>dismiss</button>
          </div>
        )}

        <div className="memory-content">
          {loading ? (
            <div className="memory-empty">
              <Loader2 size={24} className="spin" />
              <span>Loading memories...</span>
            </div>
          ) : memories.length === 0 ? (
            <div className="memory-empty">
              <div className="memory-empty-icon">
                <Brain size={28} />
              </div>
              <h2>No memories yet</h2>
              <p>
                Create memories to help the AI remember your preferences and
                important context across conversations.
              </p>
              <button
                className="memory-empty-btn"
                onClick={() => setShowForm(true)}
              >
                <Plus size={15} />
                <span>Create Memory</span>
              </button>
            </div>
          ) : (
            <div className="memory-list">
              {memories.map((memory) => {
                const typeInfo = getTypeInfo(memory.memory_type);
                return (
                  <div key={memory.id} className="memory-card">
                    {editingId === memory.id ? (
                      <div className="memory-edit-form">
                        <textarea
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          rows={3}
                          autoFocus
                        />
                        <div className="memory-edit-controls">
                          <select
                            value={editType}
                            onChange={(e) => setEditType(e.target.value)}
                          >
                            {MEMORY_TYPES.map((t) => (
                              <option key={t.value} value={t.value}>
                                {t.label}
                              </option>
                            ))}
                          </select>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={editImportance}
                            onChange={(e) =>
                              setEditImportance(parseFloat(e.target.value))
                            }
                          />
                          <span className="memory-edit-importance">
                            {editImportance.toFixed(1)}
                          </span>
                          <button
                            className="memory-edit-save"
                            onClick={() => saveEdit(memory.id)}
                          >
                            <Save size={14} /> Save
                          </button>
                          <button
                            className="memory-edit-cancel"
                            onClick={cancelEdit}
                          >
                            <X size={14} /> Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="memory-card-header">
                          <span
                            className={
                              "memory-type-badge memory-type-" + typeInfo.color
                            }
                          >
                            <Tag size={12} />
                            {typeInfo.label}
                          </span>
                          <div className="memory-card-actions">
                            <button
                              className="memory-card-edit"
                              onClick={() => startEdit(memory)}
                              title="Edit memory"
                            >
                              <Edit3 size={13} />
                            </button>
                            <button
                              className="memory-card-delete"
                              onClick={() => deleteMemory(memory.id)}
                              title="Delete memory"
                            >
                              <Trash2 size={13} />
                            </button>
                          </div>
                        </div>
                        <div className="memory-card-content">
                          {memory.content}
                        </div>
                        <div className="memory-card-footer">
                          <span className="memory-card-date">
                            <Clock size={12} />
                            {formatDate(memory.updated_at)}
                          </span>
                          <span className="memory-card-importance">
                            <Star
                              size={12}
                              className={
                                memory.importance >= 0.7 ? "star-active" : ""
                              }
                            />
                            Importance: {memory.importance.toFixed(1)}
                          </span>
                          {memory.case_id && (
                            <span className="memory-card-case">
                              Case #{memory.case_id}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Memory;
