import { api } from "./client";

/**
 * Chats API
 *
 * Backend endpoints:
 *   POST   /cases/{id}/chats       - create chat session
 *   GET    /cases/{id}/chats       - list chat sessions for a case
 *   GET    /chats/{id}             - get chat history with messages
 *   POST   /chats/{id}/messages    - send message (returns assistant response)
 *   DELETE /chats/{id}             - delete chat session
 */

export const chatsApi = {
  listSessions: (caseId) => api.get(`/cases/${caseId}/chats`),

  getHistory: (chatId) => api.get(`/chats/${chatId}`),

  createSession: (caseId, data) =>
    api.post(`/cases/${caseId}/chats`, data),

  sendMessage: (chatId, content, topK = 3) =>
    api.post(`/chats/${chatId}/messages?top_k=${topK}`, { content }),

  deleteSession: (chatId) => api.delete(`/chats/${chatId}`),
};
