import { api } from "./client";

/**
 * Cases API
 *
 * Backend endpoints:
 *   GET    /cases            - list all cases
 *   POST   /cases            - create a case
 *   GET    /cases/{id}       - get single case
 *   DELETE /cases/{id}       - delete a case
 */

export const casesApi = {
  list: () => api.get("/cases"),

  get: (caseId) => api.get(`/cases/${caseId}`),

  create: (data) => api.post("/cases", data),

  delete: (caseId) => api.delete(`/cases/${caseId}`),
};
