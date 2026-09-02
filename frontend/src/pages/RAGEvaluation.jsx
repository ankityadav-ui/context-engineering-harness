import { useEffect, useState } from "react";
import {
  Search,
  Loader2,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronRight,
  Settings2,
  BarChart3,
  Filter,
  FileText,
  ArrowDown,
  Layers,
  Zap,
  Play,
  Plus,
  Trash2,
  Target,
  Award,
  TrendingUp,
} from "lucide-react";
import { casesApi } from "../api/cases";
import { evalApi } from "../api/eval";
import { ApiError } from "../api/client";

function RAGEvaluation() {
  const [caseId, setCaseId] = useState("1");
  const [cases, setCases] = useState([]);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);
  const [expandedChunks, setExpandedChunks] = useState({});
  const [showContext, setShowContext] = useState(false);

  // Evaluation dataset state
  const [evalQueries, setEvalQueries] = useState([]);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResults, setEvalResults] = useState(null);
  const [evalTab, setEvalTab] = useState("debug");
  const [newQuery, setNewQuery] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newExpectedDoc, setNewExpectedDoc] = useState("");

  useEffect(() => {
    loadCases();
  }, []);

  useEffect(() => {
    if (evalTab === "dataset") {
      loadEvalQueries();
    }
  }, [evalTab, caseId]);

  const loadCases = async () => {
    try {
      const data = await casesApi.list();
      setCases(data);
    } catch (err) {
      console.error("Failed to load cases:", err);
    }
  };

  const loadEvalQueries = async () => {
    try {
      const data = await evalApi.listQueries(caseId);
      setEvalQueries(data);
    } catch (err) {
      console.error("Failed to load eval queries:", err);
    }
  };

  const seedEvalQueries = async () => {
    setEvalLoading(true);
    setError("");
    try {
      await evalApi.seedQueries(caseId);
      await loadEvalQueries();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message);
      }
    } finally {
      setEvalLoading(false);
    }
  };

  const addEvalQuery = async () => {
    if (!newQuery.trim() || !newExpectedDoc.trim()) return;
    setEvalLoading(true);
    setError("");
    try {
      await evalApi.createQuery(caseId, {
        query: newQuery,
        description: newDesc || null,
        expected_document_ids: [parseInt(newExpectedDoc)],
      });
      setNewQuery("");
      setNewDesc("");
      setNewExpectedDoc("");
      await loadEvalQueries();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message);
      }
    } finally {
      setEvalLoading(false);
    }
  };

  const deleteEvalQuery = async (queryId) => {
    try {
      await evalApi.deleteQuery(queryId);
      await loadEvalQueries();
    } catch (err) {
      console.error("Failed to delete query:", err);
    }
  };

  const runEvaluation = async () => {
    setEvalLoading(true);
    setError("");
    setEvalResults(null);
    try {
      const data = await evalApi.runEvaluation(caseId, topK);
      setEvalResults(data);
      await loadEvalQueries();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message);
      }
    } finally {
      setEvalLoading(false);
    }
  };

  const evaluateRetrieval = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError("");
    setResults(null);
    setExpandedChunks({});
    setShowContext(false);
    try {
      const data = await evalApi.debugSearch(caseId, query.trim(), topK);
      setResults(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.getUserMessage());
      } else {
        setError(err.message || "Failed to evaluate retrieval");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      evaluateRetrieval();
    }
  };

  const toggleChunk = (key) => {
    setExpandedChunks((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const stats = results?.statistics;

  return (
    <div className="eval-layout">
      {/* ================= SIDEBAR ================= */}

      <div className="eval-sidebar">
        <div className="eval-sidebar-header">
          <h3>RAG Evaluation</h3>
        </div>

        {/* Tab Switcher */}

        <div className="eval-tabs">
          <button className={"eval-tab " + (evalTab === "debug" ? "active" : "")}
            onClick={() => setEvalTab("debug")}>
            <Search size={14} /> Debug
          </button>
          <button className={"eval-tab " + (evalTab === "dataset" ? "active" : "")}
            onClick={() => setEvalTab("dataset")}>
            <Target size={14} /> Dataset
          </button>
        </div>

        {/* Case Selector */}

        <div className="eval-case-selector">
          <Settings2 size={14} />
          <span>Case</span>
          <select
            value={caseId}
            onChange={(event) => setCaseId(event.target.value)}
          >
            {cases.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name || "Case " + c.id}
              </option>
            ))}
            {cases.length === 0 && <option value="1">Case 1</option>}
          </select>
        </div>

        {evalTab === "debug" ? (
          <>
            {/* Query Input */}

            <div className="eval-form">
              <label>Query</label>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter a search query..."
                rows={3}
              />

              <label>Top K</label>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />

              <button className="eval-button"
                onClick={evaluateRetrieval}
                disabled={loading || !query.trim()}>
                {loading ? (
                  <><Loader2 size={16} className="spin" /> Evaluating...</>
                ) : (
                  <><Search size={16} /> Evaluate Retrieval</>
                )}
              </button>
            </div>

            {/* Quick Examples */}

            <div className="eval-examples">
              <p className="eval-examples-title">Quick Examples</p>
              <button onClick={() => { setQuery("What topics are covered in Module 1?"); setTopK(5); }}>
                What topics are covered in Module 1?
              </button>
              <button onClick={() => { setQuery("What is the population of Japan?"); setTopK(5); }}>
                What is the population of Japan?
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Dataset Management */}

            <div className="eval-form">
              <label>Add Evaluation Query</label>
              <textarea
                value={newQuery}
                onChange={(event) => setNewQuery(event.target.value)}
                placeholder="Enter evaluation query..."
                rows={2}
              />
              <input
                type="text"
                placeholder="Description (optional)"
                value={newDesc}
                onChange={(event) => setNewDesc(event.target.value)}
              />
              <input
                type="number"
                placeholder="Expected Document ID"
                value={newExpectedDoc}
                onChange={(event) => setNewExpectedDoc(event.target.value)}
              />
              <button className="eval-button"
                onClick={addEvalQuery}
                disabled={evalLoading || !newQuery.trim() || !newExpectedDoc.trim()}>
                <Plus size={16} /> Add Query
              </button>
            </div>

            <div className="eval-actions">
              <button className="eval-button secondary"
                onClick={seedEvalQueries}
                disabled={evalLoading}>
                <Zap size={16} /> Seed Defaults
              </button>
              <button className="eval-button primary"
                onClick={runEvaluation}
                disabled={evalLoading || evalQueries.length === 0}>
                {evalLoading ? (
                  <><Loader2 size={16} className="spin" /> Running...</>
                ) : (
                  <><Play size={16} /> Run Evaluation</>
                )}
              </button>
            </div>

            {/* Dataset List */}

            <div className="eval-dataset-list">
              <p className="eval-examples-title">
                Evaluation Queries ({evalQueries.length})
              </p>
              {evalQueries.length === 0 ? (
                <div className="eval-dataset-empty">
                  <Target size={20} />
                  <span>No evaluation queries yet</span>
                </div>
              ) : (
                evalQueries.map((q) => (
                  <div key={q.id} className="eval-dataset-item">
                    <div className="eval-dataset-query">
                      {q.query}
                    </div>
                    <div className="eval-dataset-meta">
                      <span>Doc: {q.expected_document_ids.join(", ")}</span>
                      {q.last_result && (
                        <span className={q.last_result.passed ? "passed" : "failed"}>
                          {q.last_result.passed ? "PASS" : "FAIL"}
                        </span>
                      )}
                      <button className="eval-delete-btn"
                        onClick={() => deleteEvalQuery(q.id)}>
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </div>

      {/* ================= MAIN ================= */}

      <div className="eval-main">
        {error && (
          <div className="eval-error">
            {error}
            <button onClick={() => setError("")}>x</button>
          </div>
        )}

        {/* Debug Tab Content */}

        {evalTab === "debug" && !results && !loading && (
          <div className="eval-empty">
            <div className="eval-empty-icon">
              <BarChart3 size={28} />
            </div>
            <h2>Context Engineering Debugger</h2>
            <p>
              Evaluate the full context engineering pipeline independently
              from Gemini. Enter a query to see each pipeline stage.
            </p>
          </div>
        )}

        {evalTab === "debug" && loading && (
          <div className="eval-loading">
            <Loader2 size={24} className="spin" />
            <span>Running context engineering pipeline...</span>
          </div>
        )}

        {/* Evaluation Results */}

        {evalTab === "dataset" && evalResults && (
          <div className="eval-results">
            <div className="eval-results-header">
              <h2><Award size={20} /> Evaluation Results</h2>
              <span className="eval-results-meta">
                Case {evalResults.case_id} | Top K: {evalResults.top_k}
              </span>
            </div>

            {/* Aggregate Metrics */}

            <div className="eval-aggregate">
              <h3><TrendingUp size={15} /> Aggregate Metrics</h3>
              <div className="eval-metrics-grid">
                <div className="eval-metric-card">
                  <div className="eval-metric-value">{evalResults.aggregate.total_queries}</div>
                  <div className="eval-metric-label">Total Queries</div>
                </div>
                <div className="eval-metric-card passed">
                  <div className="eval-metric-value">{evalResults.aggregate.total_passed}</div>
                  <div className="eval-metric-label">Passed</div>
                </div>
                <div className="eval-metric-card failed">
                  <div className="eval-metric-value">{evalResults.aggregate.total_failed}</div>
                  <div className="eval-metric-label">Failed</div>
                </div>
                <div className="eval-metric-card">
                  <div className="eval-metric-value">
                    {(evalResults.aggregate.hit_at_k * 100).toFixed(0)}%
                  </div>
                  <div className="eval-metric-label">Hit@K</div>
                </div>
                <div className="eval-metric-card">
                  <div className="eval-metric-value">
                    {(evalResults.aggregate.precision_at_k * 100).toFixed(0)}%
                  </div>
                  <div className="eval-metric-label">Precision@K</div>
                </div>
                <div className="eval-metric-card">
                  <div className="eval-metric-value">
                    {(evalResults.aggregate.recall_at_k * 100).toFixed(0)}%
                  </div>
                  <div className="eval-metric-label">Recall@K</div>
                </div>
                <div className="eval-metric-card">
                  <div className="eval-metric-value">
                    {evalResults.aggregate.mrr.toFixed(3)}
                  </div>
                  <div className="eval-metric-label">MRR</div>
                </div>
                <div className="eval-metric-card">
                  <div className="eval-metric-value">
                    {evalResults.aggregate.avg_distance?.toFixed(3) ?? "-"}
                  </div>
                  <div className="eval-metric-label">Avg Distance</div>
                </div>
              </div>
            </div>

            {/* Per-Query Results */}

            <div className="eval-query-results">
              <h3><FileText size={15} /> Per-Query Results</h3>
              {evalResults.query_results.map((qr, idx) => (
                <div key={idx} className={"eval-query-card " + (qr.passed ? "passed" : "failed")}>
                  <div className="eval-query-card-header">
                    <span className={"eval-pass-badge " + (qr.passed ? "pass" : "fail")}>
                      {qr.passed ? "PASS" : "FAIL"}
                    </span>
                    <span className="eval-query-text">{qr.query}</span>
                  </div>
                  {qr.description && (
                    <div className="eval-query-desc">{qr.description}</div>
                  )}
                  {qr.error ? (
                    <div className="eval-query-error">Error: {qr.error}</div>
                  ) : (
                    <div className="eval-query-metrics">
                      <span>Hit@K: {qr.hit_at_k ? "Yes" : "No"}</span>
                      <span>Precision@K: {qr.precision_at_k?.toFixed(3)}</span>
                      <span>Recall@K: {qr.recall_at_k?.toFixed(3)}</span>
                      <span>RR: {qr.reciprocal_rank?.toFixed(3)}</span>
                      <span>Expected: Doc {qr.expected_document_ids?.join(", ")}</span>
                      <span>Retrieved: Doc {qr.retrieved_doc_ids?.join(", ") || "none"}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {evalTab === "dataset" && !evalResults && !evalLoading && (
          <div className="eval-empty">
            <div className="eval-empty-icon">
              <Target size={28} />
            </div>
            <h2>RAG Evaluation Dataset</h2>
            <p>
              Create evaluation queries with expected document IDs,
              then run the evaluation to measure retrieval quality
              using Hit@K, Precision@K, Recall@K, and MRR.
            </p>
          </div>
        )}

        {evalTab === "dataset" && evalLoading && (
          <div className="eval-loading">
            <Loader2 size={24} className="spin" />
            <span>Running evaluation...</span>
          </div>
        )}

        {/* Debug Tab Results */}

        {evalTab === "debug" && results && (
          <div className="eval-results">
            {/* Query Info */}

            <div className="eval-query-info">
              <div className="eval-query-label">Query</div>
              <div className="eval-query-text">{results.query}</div>
              <div className="eval-query-meta">
                Case: {results.case_id} | Top K: {results.requested_top_k} |
                Threshold: {stats?.distance_threshold}
              </div>
            </div>

            {/* Pipeline Stages */}

            {stats && (
              <div className="eval-pipeline">
                <h3><Layers size={15} /> Pipeline Stages</h3>
                <div className="pipeline-flow">
                  <div className="pipeline-stage">
                    <div className="pipeline-stage-count">{stats.chunks_retrieved}</div>
                    <div className="pipeline-stage-label">Retrieved</div>
                  </div>
                  <ArrowDown size={16} className="pipeline-arrow" />
                  <div className="pipeline-stage">
                    <div className="pipeline-stage-count">{stats.chunks_after_filter}</div>
                    <div className="pipeline-stage-label">Accepted</div>
                  </div>
                  <ArrowDown size={16} className="pipeline-arrow" />
                  <div className="pipeline-stage">
                    <div className="pipeline-stage-count">{stats.chunks_after_dedup}</div>
                    <div className="pipeline-stage-label">Deduplicated</div>
                  </div>
                  <ArrowDown size={16} className="pipeline-arrow" />
                  <div className="pipeline-stage final">
                    <div className="pipeline-stage-count">{stats.chunks_used}</div>
                    <div className="pipeline-stage-label">Used in Context</div>
                  </div>
                </div>
                <div className="pipeline-meta">
                  <span>Context: {stats.context_character_count?.toLocaleString()} chars</span>
                  <span>~{stats.context_token_estimate?.toLocaleString()} tokens</span>
                  <span>Budget: {stats.max_context_chars?.toLocaleString()} chars</span>
                </div>
              </div>
            )}

            {/* Final Context */}

            {results.context && (
              <div className="eval-context-section">
                <div className="eval-context-header" onClick={() => setShowContext(!showContext)}>
                  {showContext ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                  <Zap size={15} />
                  <span>Final Context Sent to LLM</span>
                  <span className="eval-context-size">
                    {stats?.context_character_count?.toLocaleString()} chars
                  </span>
                </div>
                {showContext && (
                  <div className="eval-context-content">
                    {results.context}
                  </div>
                )}
              </div>
            )}

            {/* Final Context Chunks */}

            {results.final_context_chunks?.length > 0 && (
              <div className="eval-chunks-section">
                <h3>
                  <CheckCircle size={15} className="accepted-icon" />
                  Final Context Chunks ({results.final_context_chunks.length})
                </h3>
                <div className="eval-chunks-list">
                  {results.final_context_chunks.map((chunk, index) => {
                    const key = "final-" + index;
                    const isExpanded = expandedChunks[key];
                    return (
                      <div key={key} className="eval-chunk accepted">
                        <div className="eval-chunk-header" onClick={() => toggleChunk(key)}>
                          {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                          <CheckCircle size={14} className="accepted-icon" />
                          <span className="eval-chunk-filename">
                            {chunk.filename || "Document " + chunk.document_id}
                          </span>
                          <span className="eval-chunk-index">
                            Chunk {chunk.chunk_index}
                          </span>
                          <span className="eval-chunk-distance accepted-distance">
                            {chunk.distance.toFixed(4)}
                          </span>
                        </div>
                        {isExpanded && <div className="eval-chunk-text">{chunk.text}</div>}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* All Retrieved Chunks */}

            <div className="eval-chunks-section">
              <h3>
                <FileText size={15} />
                All Retrieved Chunks ({results.retrieved_chunks.length})
              </h3>
              {results.retrieved_chunks.length === 0 ? (
                <div className="eval-no-chunks">No chunks retrieved.</div>
              ) : (
                <div className="eval-chunks-list">
                  {results.retrieved_chunks.map((chunk, index) => {
                    const key = "all-" + index;
                    const isExpanded = expandedChunks[key];
                    const isInFinal = results.final_context_chunks?.some(
                      (f) => f.document_id === chunk.document_id && f.chunk_index === chunk.chunk_index
                    );
                    return (
                      <div key={key} className={"eval-chunk " + (isInFinal ? "accepted" : "rejected")}>
                        <div className="eval-chunk-header" onClick={() => toggleChunk(key)}>
                          {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                          {isInFinal ? <CheckCircle size={14} className="accepted-icon" /> : <XCircle size={14} className="rejected-icon" />}
                          <span className="eval-chunk-badge">
                            {isInFinal ? "USED" : "EXCLUDED"}
                          </span>
                          <span className="eval-chunk-filename">
                            {chunk.filename || "Document " + chunk.document_id}
                          </span>
                          <span className="eval-chunk-index">
                            Chunk {chunk.chunk_index}
                          </span>
                          <span className={"eval-chunk-distance " + (isInFinal ? "accepted-distance" : "rejected-distance")}>
                            {chunk.distance.toFixed(4)}
                          </span>
                        </div>
                        {isExpanded && <div className="eval-chunk-text">{chunk.text}</div>}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RAGEvaluation;

