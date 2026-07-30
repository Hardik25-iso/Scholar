import { useEffect, useRef, useState } from "react";
import { askStream, deletePaper, listPapers, uploadPaper, type Answer, type Citation, type Paper } from "../api";
import { useAuth } from "../auth/AuthContext";
import AnswerView from "../components/AnswerView";
import Library from "../components/Library";
import PdfViewer, { type ViewerTarget } from "../components/PdfViewer";
import QueryBar from "../components/QueryBar";
import SourcePanel from "../components/SourcePanel";

interface Exchange {
  question: string;
  answer: Answer | null; // set once citations arrive; text grows as it streams
  streaming: boolean;    // true while tokens are still arriving
  error: string | null;
}

// Paper-agnostic starters — they read well against whatever the user uploaded.
const SUGGESTIONS = [
  "Summarize the main contribution of this paper.",
  "What method or approach does it propose?",
  "What are the key results and their limitations?",
];

export default function Workspace() {
  const { user, logout } = useAuth();

  // ——— library state ———
  const [papers, setPapers] = useState<Paper[]>([]);
  const [papersLoading, setPapersLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [libError, setLibError] = useState<string | null>(null);

  // ——— chat state ———
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [busy, setBusy] = useState(false);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Open a citation in the PDF viewer: map its paper_id slug -> the Paper row
  // (the file endpoint keys on the DB id). If the paper was deleted, do nothing.
  const openSource = (c: Citation) => {
    const paper = papers.find((p) => p.paper_id === c.paper_id);
    if (paper) setViewer({ paperDbId: paper.id, title: paper.title, page: c.page });
  };

  const latestAnswer = [...exchanges].reverse().find((e) => e.answer)?.answer;
  const hasPapers = papers.length > 0;

  useEffect(() => {
    listPapers().then(setPapers).catch(() => {}).finally(() => setPapersLoading(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [exchanges]);

  const handleUpload = async (file: File) => {
    setLibError(null);
    setUploading(true);
    try {
      const paper = await uploadPaper(file);
      setPapers((ps) => [paper, ...ps]);
    } catch (err) {
      setLibError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deletePaper(id);
      setPapers((ps) => ps.filter((p) => p.id !== id));
    } catch (err) {
      setLibError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleAsk = async (question: string) => {
    setBusy(true);
    setActiveCitation(null);

    // Prior completed turns become history so the server can resolve follow-ups
    // into standalone questions. Strip [n] markers — they'd be noise to condense.
    const history = exchanges
      .filter((x) => x.answer && !x.streaming && !x.error)
      .map((x) => ({ question: x.question, answer: x.answer!.answer.replace(/\[\d+\]/g, "").trim() }));

    setExchanges((xs) => [...xs, { question, answer: null, streaming: true, error: null }]);

    // The busy guard means only the last exchange is ever in flight, so we
    // always update the tail. Each patch is applied to the newest exchange.
    const patch = (fn: (x: Exchange) => Exchange) =>
      setExchanges((xs) => xs.map((x, i) => (i === xs.length - 1 ? fn(x) : x)));

    try {
      await askStream(question, history, {
        onCitations: (citations) =>
          patch((x) => ({ ...x, answer: { question, answer: "", citations } })),
        onToken: (text) =>
          patch((x) => (x.answer ? { ...x, answer: { ...x.answer, answer: x.answer.answer + text } } : x)),
        onDone: () => patch((x) => ({ ...x, streaming: false })),
        onError: (message) => patch((x) => ({ ...x, streaming: false, error: message })),
      });
    } catch (err) {
      patch((x) => ({ ...x, streaming: false, error: err instanceof Error ? err.message : String(err) }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-paper text-ink">
      {/* ——— header ——— */}
      <header className="flex items-baseline justify-between border-b border-line px-6 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="font-serif text-xl font-semibold tracking-tight">Scholar</h1>
          <span className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
            grounded answers from your papers
          </span>
        </div>
        <div className="flex items-baseline gap-4">
          <span className="font-mono text-[0.62rem] text-faint">{user?.email}</span>
          <button
            onClick={() => logout()}
            className="cursor-pointer font-mono text-[0.62rem] uppercase tracking-[0.14em]
                       text-graphite transition-colors duration-200 hover:text-accent"
          >
            log out
          </button>
        </div>
      </header>

      {/* ——— three-pane workspace: library · chat · sources ——— */}
      <div className="flex min-h-0 flex-1">
        <Library
          papers={papers}
          loading={papersLoading}
          uploading={uploading}
          error={libError}
          onUpload={handleUpload}
          onDelete={handleDelete}
        />

        {/* chat pane */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-6">
            <div className="mx-auto max-w-2xl py-10">
              {exchanges.length === 0 && (
                <div className="rise mt-[16vh]">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
                    {hasPapers ? "01 — ask" : "01 — upload"}
                  </p>
                  <h2 className="mt-3 font-serif text-4xl font-medium leading-tight">
                    Ask the papers,
                    <br />
                    <em className="text-accent">not the model.</em>
                  </h2>
                  <p className="mt-4 max-w-md text-sm leading-relaxed text-graphite">
                    Every answer is assembled only from passages retrieved out of
                    your indexed papers — and cites the exact page it came from.
                  </p>

                  {hasPapers ? (
                    <div className="mt-8 space-y-2">
                      {SUGGESTIONS.map((s) => (
                        <button
                          key={s}
                          onClick={() => handleAsk(s)}
                          className="block w-full cursor-pointer border border-line bg-panel px-4 py-3
                                     text-left font-serif text-[0.95rem] text-graphite
                                     transition-colors duration-200 hover:border-accent/50 hover:text-ink"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-8 border-l-2 border-accent/50 pl-3 font-mono text-xs leading-relaxed text-graphite">
                      {papersLoading
                        ? "checking your library…"
                        : "Your library is empty. Upload a PDF on the left, then ask it anything."}
                    </p>
                  )}
                </div>
              )}

              <div className="space-y-10">
                {exchanges.map((x, i) => (
                  <section key={i}>
                    <p className="rise mb-4 border-l-2 border-accent pl-3 font-mono text-[0.8rem] text-graphite">
                      {x.question}
                    </p>
                    {x.answer && (
                      <AnswerView
                        answer={x.answer}
                        activeCitation={x.answer === latestAnswer ? activeCitation : null}
                        onCite={setActiveCitation}
                        streaming={x.streaming}
                      />
                    )}
                    {x.error && (
                      <p className="font-mono text-xs leading-relaxed text-accent">{x.error}</p>
                    )}
                    {!x.answer && !x.error && (
                      <p className="tick font-mono text-xs text-graphite">
                        retrieving · reranking · generating&nbsp;
                        <span>—</span>
                        <span>—</span>
                        <span>—</span>
                      </p>
                    )}
                  </section>
                ))}
              </div>
              <div ref={bottomRef} />
            </div>
          </div>
          <QueryBar disabled={busy || !hasPapers} onAsk={handleAsk} />
        </main>

        {/* sources pane */}
        <aside className="hidden w-[22rem] shrink-0 flex-col overflow-y-auto border-l border-line lg:flex">
          <p className="sticky top-0 border-b border-line bg-paper px-5 py-3 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
            02 — sources
          </p>
          <SourcePanel
            citations={latestAnswer?.citations ?? []}
            activeCitation={activeCitation}
            onCite={setActiveCitation}
            onOpen={openSource}
          />
        </aside>
      </div>

      {viewer && <PdfViewer target={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
