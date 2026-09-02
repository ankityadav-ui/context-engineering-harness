import { api } from "./client";

/**
 * Memory API
 *
 * Backend endpoints:
 *   POST   /memories           - create memory
 *   GET    /memories           - list memories (optional query params: user_id, case_id, memory_type)
 *   GET    /memories/{id}      - get single memory
 *   PUT    /memories/{id}      - update memory
 *   DELETE /memories/{id}      - delete memory
 */

export const memoryApi = {
  list: (params = {}) => {
    const searchParams = new URLSearchParams();
    if (params.case_id) searchParams.set("case_id", String(params.case_id));
    if (params.memory_type) searchParams.set("memory_type", params.memory_type);
    const qs = searchParams.toString();
    return api.get(`/memories${qs ? "?" + qs : ""}`);
  },

  get: (memoryId) => api.get(`/memories/${memoryId}`),

  create: (data) => api.post("/memories", data),

  update: (memoryId, data) => api.put(`/memories/${memoryId}`, data),

  delete: (memoryId) => api.delete(`/memories/${memoryId}`),
};
