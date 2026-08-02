/** Typed client for the Scholar backend — mirrors backend/models.py exactly. */

// Scholar's backend runs on 8001 (port 8000 is used by another local project).
const API_BASE = "http://localhost:8001";

export interface Citation {
  paper_id: string;
  page: number; // 0-indexed index into the document's units
  /**
   * What `page` counts in THIS document's format: only a PDF has real pages.
   * A .pptx unit is a slide, .xlsx a worksheet, .docx/.txt/.md a section.
   */
  unit: "page" | "slide" | "sheet" | "section";
  /** Ready-to-display location, e.g. "page 12" / "slide 3". Computed by the
   *  backend so the mapping lives in one place and cannot drift. */
  locator: string;
  chunk_index: number;
  score: number; // stage-1 similarity
  text: string;
  rerank_score: number | null; // stage-2 cross-encoder relevance
  /** Audit trail: the exact indexed vector, and the passage's char span within
   *  its unit. `page.slice(char_start, char_end)` is the quoted text. */
  faiss_id: number | null;
  char_start: number;
  char_end: number;
}

// ——— Audit trail ———

export interface AnswerLogSummary {
  id: number;
  created_at: string;
  question: string;
  n_citations: number;
  model: string;
  /** Whether the library is still in the state this answer was drawn from. A
   *  false here does not mean the answer was wrong — it means re-running the
   *  question today would search a different corpus. */
  reproducible: boolean;
}

export interface AnswerLogDetail extends AnswerLogSummary {
  query: string; // what retrieval ran on — differs for a condensed follow-up
  answer: string;
  citations: Citation[];
  temperature: number;
  k: number;
  candidates: number;
  papers_filter: string[] | null;
  index_fingerprint: string;
  n_chunks_indexed: number;
}

export function listAnswers(limit = 50, offset = 0): Promise<AnswerLogSummary[]> {
  return request<AnswerLogSummary[]>(`/audit?limit=${limit}&offset=${offset}`);
}

export function getAnswer(id: number): Promise<AnswerLogDetail> {
  return request<AnswerLogDetail>(`/audit/${id}`);
}

/** URL for downloading one answer with its full evidence chain. */
export function answerExportUrl(id: number, format: "json" | "csv"): string {
  return `${API_BASE}/audit/${id}/export?format=${format}`;
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

export interface ChatTurn {
  question: string;
  answer: string;
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
 *
 * `history` (prior turns) turns a bare follow-up into a standalone question on
 * the server before retrieval, so pronouns like "what about it?" still work.
 */
export async function askStream(question: string, history: ChatTurn[], h: StreamHandlers): Promise<void> {
  const csrf = readCookie("csrf_token");
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Same double-submit CSRF header the `request` helper adds. This path
      // hand-rolls fetch (it needs the raw ReadableStream), so it must set it
      // explicitly — /ask/stream requires it like every other POST.
      ...(csrf ? { "X-CSRF-Token": csrf } : {}),
    },
    credentials: "include",
    body: JSON.stringify({ question, history }),
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
/**
 * URL of a paper's stored PDF, for the in-app viewer's <iframe>. The auth
 * cookie is same-site (localhost:5173 -> localhost:8001 differ only by port, so
 * SameSite=Lax still sends it), so the iframe request is authenticated without
 * any extra work. Append "#page=N" to jump to a cited page.
 */
export function paperFileUrl(id: number): string {
  return `${API_BASE}/papers/${id}/file`;
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
