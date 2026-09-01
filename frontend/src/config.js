// Centralized API configuration
// Uses Vite env var if available, falls back to localhost
export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
