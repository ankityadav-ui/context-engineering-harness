import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// The Vite dev/preview server proxies the API prefix to the FastAPI backend,
// so the browser talks to a same-origin path and no CORS preflight is needed.
// Override the target with VITE_PROXY_TARGET for a non-default backend host.
const DEFAULT_BACKEND_TARGET = 'http://127.0.0.1:8000'

// Requests from the frontend use this prefix by default
// (see src/api/client.js and VITE_API_BASE_URL).
const API_PREFIX = '/api'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PROXY_TARGET || DEFAULT_BACKEND_TARGET

  // Strip the prefix before forwarding: the backend routes are /health,
  // /chat, /documents/upload, ... without an /api segment.
  const proxy = {
    [API_PREFIX]: {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(new RegExp(`^${API_PREFIX}`), ''),
    },
  }

  return {
    plugins: [react()],
    server: { proxy },
    preview: { proxy },
  }
})
