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

export async function ask(question: string): Promise<Answer> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}
