import { useEffect, useRef, useState } from "react";
import {
  Send,
  User,
  Sparkles,
  Settings2,
  FileText,
  Loader2,
  Plus,
  MessageSquare,
  Trash2,
  Database,
  BookOpen,
  MessageCircle,
  X,
  Cpu,
} from "lucide-react";
import { casesApi } from "../api/cases";
import { chatsApi } from "../api/chats";
import { api, ApiError } from "../api/client";

function Chats() {
  const [caseId, setCaseId] = useState("");
  const [cases, setCases] = useState([]);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [showModePicker, setShowModePicker] = useState(false);
  const [selectedMode, setSelectedMode] = useState("document");

  // LLM settings for Runtime panel
  const [llmSettings, setLlmSettings] = useState({ provider: "", model: "" });

  // Derive the active chat mode from the current session
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const activeMode = activeSession?.chat_mode || null;

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    loadCases();
    loadLlmSettings();
  }, []);

  // Listen for LLM settings changes from Settings page
  useEffect(() => {
    const handleSettingsChange = () => loadLlmSettings();
    window.addEventListener("llm-settings-changed", handleSettingsChange);
    return () => window.removeEventListener("llm-settings-changed", handleSettingsChange);
  }, []);

  useEffect(() => {
    if (caseId) {
      loadSessions();
      setActiveSessionId(null);
      setMessages([]);
    }
  }, [caseId]);

  const loadCases = async () => {
    try {
      const data = await casesApi.list();
      setCases(data);
      if (data.length > 0 && !caseId) {
        setCaseId(String(data[0].id));
      }
    } catch (err) {
      console.error("Failed to load cases:", err);
    }
  };

  const loadLlmSettings = async () => {
    try {
      const data = await api.get("/settings/llm");
      setLlmSettings(data);
    } catch (err) {
      console.error("Failed to load LLM settings:", err);
    }
  };

  useEffect(() => {
    if (activeSessionId) {
      loadChatHistory(activeSessionId);
    }
  }, [activeSessionId]);

  // Sync selectedMode when switching sessions
  useEffect(() => {
    if (activeSession) {
      setSelectedMode(activeSession.chat_mode || "document");
    }
  }, [activeSessionId]);

  const loadSessions = async () => {
    setSessionsLoading(true);
    try {
      const data = await chatsApi.listSessions(caseId);
      setSessions(data);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setSessionsLoading(false);
    }
  };

  const loadChatHistory = async (chatId) => {
    setLoading(true);
    setError("");
    try {
      const data = await chatsApi.getHistory(chatId);
      const loadedMessages = data.messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
        sources: [],
      }));
      setMessages(loadedMessages);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to load chat history");
      }
    } finally {
      setLoading(false);
    }
  };

  const createNewSession = async (mode) => {
    setError("");
    setShowModePicker(false);
    try {
      const newSession = await chatsApi.createSession(caseId, {
        title: "New Chat",
        chat_mode: mode,
      });
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      setMessages([]);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to create chat session");
      }
    }
  };

  const deleteSession = async (sessionId, event) => {
    event.stopPropagation();
    setError("");
    try {
      await chatsApi.deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to delete chat session");
      }
    }
  };

  const sendMessage = async (event, overrideQuery, autoMode) => {
    if (event?.preventDefault) event.preventDefault();
    const userMessage = (overrideQuery || query).trim();
    if (!userMessage || loading) return;

    // Use explicit autoMode if provided (e.g. from suggestion buttons),
    // otherwise fall back to selectedMode.
    const effectiveMode = autoMode || selectedMode;

    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      try {
        const newSession = await chatsApi.createSession(caseId, {
          title: userMessage.substring(0, 50) + (userMessage.length > 50 ? "..." : ""),
          chat_mode: effectiveMode,
        });
        setSessions((prev) => [newSession, ...prev]);
        currentSessionId = newSession.id;
        setActiveSessionId(currentSessionId);
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.getUserMessage());
        } else {
          setError(err.message || "Failed to create chat session");
        }
        return;
      }
    }

    setMessages((previous) => [...previous, { role: "user", content: userMessage }]);
    setQuery("");
    setError("");
    setLoading(true);

    try {
      const data = await chatsApi.sendMessage(currentSessionId, userMessage);
      setMessages((previous) => [...previous, { role: "assistant", content: data.content, sources: data.sources || [], chunk_count: data.chunk_count || 0, metadata: data.metadata || null }]);
      setSessions((prev) =>
        prev.map((s) => (s.id === currentSessionId ? { ...s, updated_at: new Date().toISOString() } : s))
      );
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else if (err instanceof TypeError && err.message.includes("fetch")) {
        setError("Unable to connect to the backend. Please check if the server is running.");
      } else {
        setError(err.message || "Unable to connect to the backend.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(event);
    }
  };

  return (
    <div className="chat-layout">
      <div className="chat-sidebar">
        <div className="chat-sidebar-header">
          <h3>Chats</h3>
        </div>
        <button className="new-chat-button" onClick={() => setShowModePicker(true)}>
          <Plus size={16} />
          <span>New Chat</span>
        </button>
        {showModePicker && (
          <div className="mode-picker">
            <div className="mode-picker-header">
              <span>Choose Chat Type</span>
              <button className="mode-picker-close" onClick={() => setShowModePicker(false)}>
                <X size={14} />
              </button>
            </div>
            <button className="mode-option" onClick={() => createNewSession("normal")}>
              <MessageCircle size={18} />
              <div className="mode-option-text">
                <span className="mode-option-title">Normal Chat</span>
                <span className="mode-option-desc">Chat directly with the configured LLM</span>
              </div>
            </button>
            <button className="mode-option" onClick={() => createNewSession("document")}>
              <BookOpen size={18} />
              <div className="mode-option-text">
                <span className="mode-option-title">Document Chat</span>
                <span className="mode-option-desc">Ask questions about your case documents</span>
              </div>
            </button>
          </div>
        )}
        <div className="chat-sessions-list">
          {sessionsLoading ? (
            <div className="sessions-loading">
              <Loader2 size={16} className="spin" />
              <span>Loading...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="sessions-empty">
              <MessageSquare size={16} />
              <span>No chats yet</span>
            </div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={"chat-session-item " + (activeSessionId === session.id ? "active" : "")}
                onClick={() => setActiveSessionId(session.id)}
              >
                {session.chat_mode === "normal" ? <MessageCircle size={14} /> : <MessageSquare size={14} />}
                <span className="session-title">{session.title}</span>
                <button
                  className="session-delete"
                  onClick={(e) => deleteSession(session.id, e)}
                  title="Delete chat"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))
          )}
        </div>
        <div className="chat-sidebar-footer">
          <div className="sidebar-case-section">
            <span className="sidebar-case-label">Current Case</span>
            <div className="case-selector">
              <Settings2 size={14} />
              <select value={caseId} onChange={(event) => setCaseId(event.target.value)}>
                {cases.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="chat-main">
        <div className="chat-topbar">
          <div>
            <h1>
              {activeSessionId
                ? sessions.find((s) => s.id === activeSessionId)?.title || "Chat"
                : "Chats"}
            </h1>
            <div className="chat-topbar-meta">
              {activeSessionId && activeMode && (
                <>
                  {activeMode === "document" && (
                    <span className="chat-topbar-case">
                      <Database size={12} />
                      {cases.find((c) => String(c.id) === String(caseId))?.name || "Case " + caseId}
                    </span>
                  )}
                  <span className={`chat-topbar-type${activeMode === "normal" ? " normal-mode" : ""}`}>
                    {activeMode === "normal" ? <MessageCircle size={12} /> : <BookOpen size={12} />}
                    {activeMode === "normal" ? "Normal Chat" : "Document Chat"}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="chat-container">
          {messages.length === 0 && !loading ? (
            <div className="empty-chat">
              <div className="empty-chat-icon">
                <Sparkles size={28} />
              </div>
              <h2>No chat selected</h2>
              {activeMode ? (
                <>
                  <p className="empty-chat-case">
                    <Database size={13} />
                    Case: {cases.find((c) => String(c.id) === String(caseId))?.name || "Case " + caseId}
                  </p>
                  <p>{activeMode === "normal"
                    ? "Send a message to start a conversation with the configured LLM."
                    : "Send a message to search your case documents."}
                  </p>
                  <div className="suggested-questions">
                    {activeMode === "normal" ? (
                      <>
                        <button onClick={() => sendMessage(null, "Hello, who are you?")}>
                          Hello, who are you?
                        </button>
                        <button onClick={() => sendMessage(null, "Explain context engineering.")}>
                          Explain context engineering.
                        </button>
                      </>
                    ) : (
                      <>
                        <button onClick={() => sendMessage(null, "What topics are covered in Module 1?")}>
                          What topics are covered in Module 1?
                        </button>
                        <button onClick={() => sendMessage(null, "Summarize the uploaded document.")}>
                          Summarize the uploaded document.
                        </button>
                      </>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <p className="empty-chat-case">
                    <Database size={13} />
                    Case: {cases.find((c) => String(c.id) === String(caseId))?.name || "Case " + caseId}
                  </p>
                  <p>Start a new chat to begin a conversation.</p>
                  <div className="suggested-questions">
                    <button onClick={() => sendMessage(null, "Hello, who are you?", "normal")}>
                      Hello, who are you?
                    </button>
                    <button onClick={() => sendMessage(null, "What topics are covered in Module 1?", "document")}>
                      What topics are covered in Module 1?
                    </button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="messages">
              {messages.map((message, index) => (
                <div key={index} className={"chat-message " + message.role}>
                  <div className="message-avatar">
                    {message.role === "user" ? <User size={17} /> : <Sparkles size={17} />}
                  </div>
                  <div className="message-body">
                    <div className="message-header">
                      <span>{message.role === "user" ? "You" : "Harness"}</span>
                    </div>
                    <div className="message-content">{message.content}</div>
                    {message.role === "assistant" && message.metadata && (
                      <div className="context-indicator">
                        {message.metadata.long_term_memory_used && (
                          <span className="context-badge memory">Memory</span>
                        )}
                        {message.metadata.document_chunks_used > 0 && (
                          <span className="context-badge document">Docs ({message.metadata.document_chunks_used})</span>
                        )}
                        {message.metadata.graph_results_used > 0 && (
                          <span className="context-badge graph">Graph ({message.metadata.graph_results_used})</span>
                        )}
                        {message.metadata.short_term_messages > 0 && (
                          <span className="context-badge history">History ({message.metadata.short_term_messages})</span>
                        )}
                      </div>
                    )}
                    {message.role === "assistant" && message.sources?.length > 0 && (
                      <div className="sources">
                        <div className="sources-title">
                          <FileText size={15} />
                          <span>Retrieved context</span>
                        </div>
                        {message.sources.map((source, sourceIndex) => (
                          <div key={sourceIndex} className="source">
                            <div className="source-top">
                              <span>{source.filename || 'Document ' + source.document_id}</span>
                              <span>Chunk {source.chunk_index}</span>
                            </div>
                            {source.distance !== undefined && (
                              <span className="distance">
                                Relevance: {(source.distance).toFixed(4)}
                              </span>
                            )}
                            {source.text && <p>{source.text}</p>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="chat-message assistant">
                  <div className="message-avatar">
                    <Sparkles size={17} />
                  </div>
                  <div className="message-body">
                    <div className="message-header">
                      <span>Harness</span>
                    </div>
                    <div className="thinking">
                      <Loader2 size={17} className="spin" />
                      <span>{activeMode === "normal" ? "Thinking..." : "Searching documents and thinking..."}</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {error && <div className="chat-error">{error}</div>}

        <div className="chat-input-wrapper">
          <form className="chat-input" onSubmit={sendMessage}>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={activeMode === "normal" ? "Ask anything..." : "Ask something about your documents..."}
              disabled={loading}
              rows={1}
            />
            <button type="submit" disabled={loading || !query.trim()} title="Send message">
              {loading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
            </button>
          </form>
          <div className="chat-disclaimer">
            Harness can make mistakes. Verify important information.
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chats;
