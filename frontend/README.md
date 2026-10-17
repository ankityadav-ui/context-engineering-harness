# Frontend (React + Vite)

> **Status: stale.** The React sources this document describes (`src/`,
> `src/features/`, `src/api/`) have been removed. The frontend is being
> rebuilt as a Streamlit app (`app.py`, `pages/`, `api_client.py`,
> `ui_theme.py`). Treat the notes below as historical until this file is
> rewritten. Note that the `/providers*` routes it lists do not exist in the
> backend; generation is Ollama-only and `GET /capabilities` reports the
> active model.

The workspace UI for the Context Engineering Harness. It is plain
JavaScript React (no TypeScript) built with Vite, and it talks to the FastAPI
backend through a single HTTP client.

## Pages

| Page | Source | Backend routes |
| --- | --- | --- |
| Chat | `src/features/chat/` | `POST /chat/stream`, `POST /masking/text`, `GET/POST /providers*` |
| Documents | `src/features/documents/` | `POST /documents/upload`, `GET /documents`, `GET /documents/{id}`, `POST /documents/search`, `POST /documents/ask` |
| Context engine | `src/features/context/` | `POST /context/preview`, `GET /documents`, `GET /conversations` |
| Knowledge graph | `src/features/knowledge-graph/` | `GET /knowledge-graph/entities*` |
| Masking | `src/features/masking/` | `POST /masking/text` |
| Settings | `src/features/settings/` | `GET /providers`, `GET /capabilities`, `GET/PUT /settings/system-prompt`, `GET /health` |

The chat page stays mounted when you switch pages so the transcript is not
lost; the other pages mount on demand and load their data when opened.

## Layout

```text
src/
  api/client.js        fetch wrapper, SSE reader, error normalization
  api/endpoints.js     one typed-thin wrapper per backend route
  components/          sidebar, provider selector, inline icon set
  features/<page>/     one directory per destination (component + CSS)
  styles/              design tokens (index.css) and shell layout (app.css)
```

Design tokens (colors, radii, shadows, typography) live in
`src/styles/index.css`; shared element styles (`.card`, `.alert`, `.button`)
are defined there too, so a page only ships the rules it adds.

## Backend API configuration

Everything goes through `src/api/client.js`. Its base URL comes from
`VITE_API_BASE_URL` and defaults to `/api`. Copy `.env.example` to `.env` to
override it; never put backend secrets there, because anything prefixed with
`VITE_` is public in the bundle.

In development and preview, Vite proxies `/api/*` to the backend
(`VITE_PROXY_TARGET`, default `http://127.0.0.1:8000`) and strips the `/api`
prefix, so the browser makes same-origin requests and no local CORS setup is
needed.

Run the backend, then the frontend:

```bash
# terminal 1 (repository root)
cd backend && python -m uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm install && npm run dev
```

Scripts: `npm run dev`, `npm run build`, `npm run preview`, `npm run lint`.

## UI verification

`node tests/ui-verify.mjs` starts the contract mock backend
(`tests/mock-documents-backend.mjs`) plus a Vite dev server and drives the app
in headless Chrome over the DevTools protocol. It needs Chrome installed (or
`CHROME_PATH`). Use `UI_VERIFY_ONLY=chat,settings` to run a subset of the
scenarios.

The mock mirrors the confirmed backend response shapes, including
`/providers`, `/capabilities`, and `/context/preview`; it is never imported by
the application.

## Notes

- Assistant messages render fenced code blocks and inline code. This is
  deliberately not a full Markdown renderer, and no HTML from the model is
  injected (everything is built as React elements).
- The React Compiler is not enabled because of its effect on dev and build
  performance.
