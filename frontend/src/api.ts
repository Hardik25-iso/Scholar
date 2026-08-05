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

// ——— Workspaces ———

export interface Workspace {
  id: number;
  name: string;
  /** A personal library. Cannot be shared or left. */
  is_personal: boolean;
  /** The CALLER's role in it — not a property of the workspace itself. */
  role: "owner" | "member";
  /** Whether this is the workspace the caller's requests currently act on. */
  is_current: boolean;
  created_at: string;
}

export interface Member {
  user_id: number;
  email: string;
  role: "owner" | "member";
  joined_at: string;
}

export interface CreatedInvitation {
  id: number;
  email: string;
  role: "owner" | "member";
  expires_at: string;
  /** Whether the server actually emailed the invitation. False means no mail
   *  provider is configured, not that anything failed silently. */
  delivered: boolean;
  /** Present ONLY when `delivered` is false — then handing it to the inviter is
   *  the only way it can reach the invitee. Null once mail is configured: an
   *  invitation token is a credential, and it should travel one route only. */
  token: string | null;
}

export function listWorkspaces(): Promise<Workspace[]> {
  return request<Workspace[]>("/workspaces");
}
export function createWorkspace(name: string): Promise<Workspace> {
  return request<Workspace>("/workspaces", { method: "POST", body: JSON.stringify({ name }) });
}
/** Point this account's requests at a different workspace. */
export function activateWorkspace(id: number): Promise<Workspace> {
  return request<Workspace>(`/workspaces/${id}/activate`, { method: "POST" });
}
export function listMembers(id: number): Promise<Member[]> {
  return request<Member[]>(`/workspaces/${id}/members`);
}
export function inviteMember(
  id: number,
  email: string,
  role: "owner" | "member" = "member",
): Promise<CreatedInvitation> {
  return request<CreatedInvitation>(`/workspaces/${id}/invitations`, {
    method: "POST",
    body: JSON.stringify({ email, role }),
  });
}
export function removeMember(workspaceId: number, userId: number): Promise<void> {
  return request<void>(`/workspaces/${workspaceId}/members/${userId}`, { method: "DELETE" });
}
export function acceptInvitation(token: string): Promise<Workspace> {
  return request<Workspace>("/workspaces/accept", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
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
  query: string; // the standalone question after condensing; what generation saw
  /** What retrieval and reranking ACTUALLY ran on, when query expansion changed
   *  it — the user's question plus an LLM-generated hypothetical answer. null
   *  means no expansion happened, not that it was withheld. Recorded because
   *  expansion makes retrieval non-deterministic, and `reproducible` would
   *  otherwise be claiming more than it can support. */
  retrieval_query: string | null;
  expansion_mode: "none" | "prf" | "hyde";
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
function buildHeaders(options: RequestInit): Headers {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  if (options.body) headers.set("Content-Type", "application/json");
  if (method !== "GET" && method !== "HEAD") {
    const csrf = readCookie("csrf_token");
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  return headers;
}

/**
 * Ask the backend for a new access token using the refresh cookie.
 *
 * Shared across concurrent callers: a page that fires several requests at once
 * would otherwise hit 401 on each and start a refresh per request, and all but
 * one would be wasted. The in-flight promise is reused instead.
 */
let refreshing: Promise<boolean> | null = null;

function refreshSession(): Promise<boolean> {
  if (!refreshing) {
    refreshing = fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: buildHeaders({ method: "POST" }),
      credentials: "include",
    })
      .then((r) => r.ok)
      .catch(() => false)
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

async function request<T>(path: string, options: RequestInit = {}, retry = true): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: buildHeaders(options),
    credentials: "include", // send/receive the auth cookie cross-origin
  });

  // The access token is short-lived by design. Rather than logging someone out
  // mid-sentence, spend one refresh and replay the request. Only ever once —
  // if the refresh token is gone too, the session really has ended, and a loop
  // here would hammer the server on every failed request.
  // `/auth/refresh` itself is excluded so a failed refresh cannot recurse.
  if (res.status === 401 && retry && path !== "/auth/refresh") {
    if (await refreshSession()) return request<T>(path, options, false);
  }

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
/** An upload in progress. Indexing happens off the request, so this — not a
 *  Paper — is what an upload returns. */
export interface IndexJob {
  id: number;
  paper_id: string;
  filename: string;
  title: string;
  status: "queued" | "running" | "done" | "failed";
  /** Set when status is "failed": the reason the upload could not be indexed. */
  error: string | null;
  n_chunks: number;
  /** True when no queue was reachable and the work ran inside the request. */
  ran_inline: boolean;
  created_at: string;
}

export function getIndexJob(id: number): Promise<IndexJob> {
  return request<IndexJob>(`/papers/jobs/${id}`);
}

/** Uploads still being indexed in this workspace — lets a reloaded page resume. */
export function listIndexJobs(): Promise<IndexJob[]> {
  return request<IndexJob[]>("/papers/jobs");
}

/**
 * Upload a document. Uses FormData (multipart) rather than the JSON `request`
 * helper, so we must NOT set Content-Type — the browser sets it with the
 * multipart boundary. Still sends the auth cookie + the CSRF header.
 *
 * Returns a JOB, not a Paper: indexing runs outside the request so a large
 * document cannot exceed the proxy timeout. Poll `getIndexJob` until its status
 * is terminal — and note the job may ALREADY be terminal here, when the server
 * has no queue and indexed inline.
 */
export async function uploadPaper(file: File): Promise<IndexJob> {
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
/** Download everything this account holds. A plain link, so the browser handles
 *  the download; the auth cookie is same-site and travels with it. */
export function accountExportUrl(): string {
  return `${API_BASE}/auth/export`;
}
/** Irreversible. The password is required again by the server. */
export function deleteAccount(password: string): Promise<{ status: string }> {
  return request("/auth/delete", { method: "POST", body: JSON.stringify({ password }) });
}
/**
 * Begin a password reset. Always resolves, whether or not the account exists —
 * the backend answers 202 either way so the endpoint cannot be used to test
 * whether someone has an account here. The UI must show the same message.
 */
export function forgotPassword(email: string): Promise<{ status: string }> {
  return request("/auth/forgot", { method: "POST", body: JSON.stringify({ email }) });
}
/** Consume a reset token and set a new password. Logs the user straight in. */
export function resetPassword(token: string, password: string): Promise<User> {
  return request<User>("/auth/reset", {
    method: "POST",
    body: JSON.stringify({ token, password }),
  });
}
