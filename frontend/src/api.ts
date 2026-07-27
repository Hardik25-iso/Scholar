/** Typed client for the Scholar backend — mirrors backend/models.py exactly. */

// Scholar's backend runs on 8001 (port 8000 is used by another local project).
const API_BASE = "http://localhost:8001";

export interface Citation {
  paper_id: string;
  page: number; // 0-indexed; display as page + 1
  chunk_index: number;
  score: number; // stage-1 cosine similarity
  text: string;
  rerank_score: number | null; // stage-2 cross-encoder relevance
}

export interface Answer {
  question: string;
  answer: string;
  citations: Citation[];
}

export interface User {
  id: number;
  email: string;
}

export interface Paper {
  id: number;
  paper_id: string;
  title: string;
  filename: string;
  n_chunks: number;
  created_at: string;
}

/** Raised for non-2xx responses; carries the status so callers can branch. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Read a non-httpOnly cookie by name (used for the CSRF token). */
function readCookie(name: string): string | null {
  const m = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return m ? decodeURIComponent(m[1]) : null;
}

/**
 * Wrapper around fetch that always sends cookies (credentials: "include" — the
 * auth cookie is cross-origin 5173->8001) and attaches the CSRF header on
 * unsafe methods via the double-submit pattern.
 */
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  if (options.body) headers.set("Content-Type", "application/json");
  if (method !== "GET" && method !== "HEAD") {
    const csrf = readCookie("csrf_token");
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: "include", // send/receive the auth cookie cross-origin
  });
  if (!res.ok) await raise(res);
  if (res.status === 204) return undefined as T; // No Content (e.g. DELETE)
  return res.json();
}

/** Parse an error response body and throw a typed ApiError. */
async function raise(res: Response): Promise<never> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail ?? detail;
    // Pydantic 422 returns a list of field errors; surface the first message.
    if (Array.isArray(detail)) detail = detail[0]?.msg ?? "invalid input";
  } catch {
    /* non-JSON error body */
  }
  throw new ApiError(res.status, detail);
}

// ——— RAG ———
export function ask(question: string): Promise<Answer> {
  return request<Answer>("/ask", { method: "POST", body: JSON.stringify({ question }) });
}

export interface StreamHandlers {
  onCitations: (citations: Citation[]) => void; // sent first, before any text
  onToken: (text: string) => void;              // one incremental answer delta
  onDone: () => void;                            // stream finished cleanly
  onError: (message: string) => void;           // mid-stream generation failure
}

/**
 * Streaming ask: retrieval + reranking happen up front (citations arrive first),
 * then the answer streams token-by-token. Uses fetch + a ReadableStream reader
 * parsing NDJSON — EventSource can't send our cross-origin auth cookie on a POST.
 */
export async function askStream(question: string, h: StreamHandlers): Promise<void> {
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) return raise(res); // raise() throws (never returns)

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // NDJSON: dispatch each complete line, keep any partial tail in the buffer.
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.type === "citations") h.onCitations(msg.citations);
      else if (msg.type === "token") h.onToken(msg.text);
      else if (msg.type === "done") h.onDone();
      else if (msg.type === "error") h.onError(msg.detail);
    }
  }
}

// ——— Library ———
export function listPapers(): Promise<Paper[]> {
  return request<Paper[]>("/papers");
}
export function deletePaper(id: number): Promise<void> {
  return request<void>(`/papers/${id}`, { method: "DELETE" });
}
/**
 * Upload a PDF. Uses FormData (multipart) rather than the JSON `request`
 * helper, so we must NOT set Content-Type — the browser sets it with the
 * multipart boundary. Still sends the auth cookie + the CSRF header.
 */
export async function uploadPaper(file: File): Promise<Paper> {
  const form = new FormData();
  form.append("file", file);
  const csrf = readCookie("csrf_token");
  const res = await fetch(`${API_BASE}/papers`, {
    method: "POST",
    body: form,
    credentials: "include",
    headers: csrf ? { "X-CSRF-Token": csrf } : {},
  });
  if (!res.ok) await raise(res);
  return res.json();
}

// ——— Auth ———
export function register(email: string, password: string): Promise<User> {
  return request<User>("/auth/register", { method: "POST", body: JSON.stringify({ email, password }) });
}
export function login(email: string, password: string): Promise<User> {
  return request<User>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
}
export function logout(): Promise<{ status: string }> {
  return request("/auth/logout", { method: "POST" });
}
export function me(): Promise<User> {
  return request<User>("/auth/me");
}
