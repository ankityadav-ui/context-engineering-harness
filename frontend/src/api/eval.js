import { api } from "./client";

/**
 * RAG Evaluation API
 *
 * Backend endpoints:
 *   GET    /cases/{id}/search/debug         - debug search (pipeline inspection)
 *   GET    /cases/{id}/search               - semantic search
 *   GET    /cases/{id}/eval-queries         - list eval queries
 *   POST   /cases/{id}/eval-queries         - create eval query
 *   POST   /cases/{id}/eval-queries/seed    - seed default eval queries
 *   DELETE /eval-queries/{id}               - delete eval query
 *   POST   /cases/{id}/evaluation/run       - run evaluation suite
 */

export const evalApi = {
  debugSearch: (caseId, query, topK = 5) => {
    const params = new URLSearchParams({ query, top_k: String(topK) });
    return api.get(`/cases/${caseId}/search/debug?${params.toString()}`);
  },

  search: (caseId, query, topK = 5) => {
    const params = new URLSearchParams({ query, top_k: String(topK) });
    return api.get(`/cases/${caseId}/search?${params.toString()}`);
  },

  listQueries: (caseId) => api.get(`/cases/${caseId}/eval-queries`),

  createQuery: (caseId, data) =>
    api.post(`/cases/${caseId}/eval-queries`, data),

  seedQueries: (caseId) =>
    api.post(`/cases/${caseId}/eval-queries/seed`),

  deleteQuery: (queryId) => api.delete(`/eval-queries/${queryId}`),

  runEvaluation: (caseId, topK = 5) =>
    api.post(`/cases/${caseId}/evaluation/run?top_k=${topK}`),
};
