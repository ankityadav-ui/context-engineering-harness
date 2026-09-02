import { apiFetch } from "./client";
import { api } from "./client";

/**
 * Documents API
 *
 * Backend endpoints:
 *   GET    /cases/{id}/documents                    - list documents for a case
 *   POST   /cases/{id}/documents                    - upload document (multipart)
 *   GET    /cases/{id}/documents/{docId}            - get document details
 *   GET    /cases/{id}/documents/{docId}/chunks      - get document chunks
 *   DELETE /cases/{id}/documents/{docId}             - delete document
 */

export const documentsApi = {
  list: (caseId) => api.get(`/cases/${caseId}/documents`),

  upload: (caseId, file) => {
    const formData = new FormData();
    formData.append("file", file);
    // Use apiFetch directly since FormData should not be JSON-stringified
    return apiFetch(`/cases/${caseId}/documents`, {
      method: "POST",
      body: formData,
    });
  },

  getDetails: (caseId, docId) =>
    api.get(`/cases/${caseId}/documents/${docId}`),

  getChunks: (caseId, docId) =>
    api.get(`/cases/${caseId}/documents/${docId}/chunks`),

  delete: (caseId, docId) =>
    api.delete(`/cases/${caseId}/documents/${docId}`),
};
