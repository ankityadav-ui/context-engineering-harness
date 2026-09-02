import { API_URL } from "../config";

/**
 * Custom error class for API errors with user-friendly messages.
 */
export class ApiError extends Error {
  constructor(status, detail, rawError) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.rawError = rawError;
  }

  /**
   * Get a user-friendly error message based on HTTP status.
   */
  getUserMessage() {
    switch (this.status) {
      case 400:
        return this.message || "Invalid request. Please check your input.";
      case 404:
        return "The requested resource was not found.";
      case 429:
        return "Too many requests. Please wait a moment and try again.";
      case 500:
        return "A server error occurred. Please try again later.";
      case 502:
        return "The AI service is temporarily unavailable. Please try again.";
      case 504:
        return "The request timed out. Please try again.";
      default:
        return this.message || "An unexpected error occurred.";
    }
  }
}

/**
 * Centralized fetch wrapper for all API calls.
 *
 * - Prepends API_URL
 * - Sets Content-Type for JSON
 * - Handles non-OK responses with structured errors
 * - Returns parsed JSON
 */
export async function apiFetch(path, options = {}) {
  const { body, method = "GET", headers: extraHeaders = {}, ...rest } = options;

  const headers = { ...extraHeaders };

  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const fetchOptions = {
    method,
    headers,
    ...rest,
  };

  if (body) {
    fetchOptions.body = body instanceof FormData ? body : JSON.stringify(body);
  }

  const response = await fetch(`${API_URL}${path}`, fetchOptions);

  // Try to parse JSON regardless of status
  let data;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || `HTTP ${response.status}`;
    throw new ApiError(response.status, detail, data);
  }

  return data;
}

/**
 * Convenience wrappers for common HTTP methods.
 */
export const api = {
  get: (path, options) => apiFetch(path, { method: "GET", ...options }),
  post: (path, body, options) => apiFetch(path, { method: "POST", body, ...options }),
  put: (path, body, options) => apiFetch(path, { method: "PUT", body, ...options }),
  delete: (path, options) => apiFetch(path, { method: "DELETE", ...options }),
};
