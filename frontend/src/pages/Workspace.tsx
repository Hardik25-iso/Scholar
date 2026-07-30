import { useEffect, useRef, useState } from "react";
import { askStream, deletePaper, listPapers, uploadPaper, type Answer, type Citation, type Paper } from "../api";
import { useAuth } from "../auth/AuthContext";
import AnswerView from "../components/AnswerView";
import { AskIcon, LayersIcon, LogoutIcon } from "../components/icons";
import Library from "../components/Library";
import PdfViewer, { type ViewerTarget } from "../components/PdfViewer";
import QueryBar from "../components/QueryBar";
import SourcePanel from "../components/SourcePanel";

// Warm radial wash behind the whole workspace — lifts it off a flat fill.
const PAGE_BG = "radial-gradient(120% 80% at 50% -10%, #FBFAF6 0%, #F7F5F0 55%)";

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
    <div className="flex h-screen flex-col text-ink" style={{ background: PAGE_BG }}>
      {/* ——— top bar ——— */}
      <header className="relative z-10 flex items-center justify-between border-b border-line
                         bg-panel px-5 py-[0.7rem] shadow-e1">
        <div className="flex items-baseline gap-[0.6rem]">
          <span className="h-[0.4rem] w-[0.4rem] self-center rounded-full bg-accent" />
          <h1 className="font-serif text-[1.28rem] font-semibold tracking-[-0.01em]">Scholar</h1>
          <span className="hidden font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint sm:inline">
            grounded answers from your papers
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="hidden font-mono text-[0.66rem] text-graphite sm:inline">{user?.email}</span>
          <button
            onClick={() => logout()}
            className="inline-flex cursor-pointer items-center gap-[0.35rem] font-mono text-[0.62rem]
                       uppercase tracking-[0.12em] text-graphite transition-colors duration-200 hover:text-accent"
          >
            <LogoutIcon className="h-[0.85rem] w-[0.85rem]" /> log out
          </button>
        </div>
      </header>

      {/* ——— three-pane workspace: library · chat · sources ——— */}
      <div className="relative z-[1] flex min-h-0 flex-1">
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
          <div className="flex-1 overflow-y-auto px-2">
            <div className="mx-auto max-w-[44rem] px-6 py-8">
              {exchanges.length === 0 && (
                <div className="rise mt-[14vh]">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
                    {hasPapers ? "01 — ask" : "01 — upload"}
                  </p>
                  <h2 className="mt-3 font-serif text-[2.6rem] font-medium leading-[1.05]">
                    Ask the papers,
                    <br />
                    <em className="text-accent">not the model.</em>
                  </h2>
                  <p className="mt-4 max-w-md text-[0.95rem] leading-relaxed text-graphite">
                    Every answer is assembled only from passages retrieved out of
                    your indexed papers — and cites the exact page it came from.
                  </p>

                  {hasPapers ? (
                    <div className="mt-8 space-y-[0.55rem]">
                      {SUGGESTIONS.map((s) => (
                        <button
                          key={s}
                          onClick={() => handleAsk(s)}
                          className="block w-full cursor-pointer rounded-[10px] border border-lineSoft bg-raised
                                     px-4 py-3 text-left font-serif text-[0.95rem] text-graphite shadow-e1
                                     transition-all duration-200 hover:-translate-y-0.5 hover:text-ink hover:shadow-e2"
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

              <div className="flex flex-col gap-[1.6rem]">
                {exchanges.map((x, i) => {
                  // A turn is a follow-up if an earlier turn was already answered.
                  const isFollowUp = exchanges.slice(0, i).some((e) => e.answer && !e.error);
                  return (
                    <section key={i} className="rise flex flex-col gap-[0.7rem]">
                      <div className="flex items-start gap-[0.6rem]">
                        <span className="grid h-[1.35rem] w-[1.35rem] shrink-0 place-items-center rounded-full bg-ink text-paper">
                          <AskIcon className="h-[0.8rem] w-[0.8rem]" />
                        </span>
                        <p className="mt-[0.15rem] font-serif text-[1.02rem] italic text-graphite">{x.question}</p>
                        {isFollowUp && (
                          <span className="mt-[0.2rem] shrink-0 self-center rounded-full border border-accentSoft
                                           bg-[#FCF4F2] px-[0.4rem] py-[0.1rem] font-mono text-[0.55rem]
                                           uppercase tracking-[0.14em] text-accent">
                            follow-up
                          </span>
                        )}
                      </div>

                      {x.answer && (
                        <div className="rounded-[14px] border border-lineSoft bg-panel px-[1.3rem] py-[1.15rem] shadow-e2">
                          <AnswerView
                            answer={x.answer}
                            activeCitation={x.answer === latestAnswer ? activeCitation : null}
                            onCite={setActiveCitation}
                            streaming={x.streaming}
                          />
                          {!x.streaming && (
                            <div className="mt-[0.9rem] border-t border-lineSoft pt-[0.7rem] font-mono text-[0.62rem] text-faint">
                              grounded in {x.answer.citations.length} passages
                            </div>
                          )}
                        </div>
                      )}
                      {x.error && (
                        <p className="font-mono text-xs leading-relaxed text-accent">{x.error}</p>
                      )}
                      {!x.answer && !x.error && (
                        <p className="tick pl-[1.95rem] font-mono text-xs text-graphite">
                          retrieving · reranking · generating&nbsp;
                          <span>—</span>
                          <span>—</span>
                          <span>—</span>
                        </p>
                      )}
                    </section>
                  );
                })}
              </div>
              <div ref={bottomRef} />
            </div>
          </div>
          <QueryBar disabled={busy || !hasPapers} onAsk={handleAsk} />
        </main>

        {/* sources pane */}
        <aside className="hidden w-[344px] shrink-0 flex-col overflow-y-auto border-l border-line lg:flex">
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line
                          bg-paper px-[1.15rem] py-[0.85rem]">
            <span className="flex items-center gap-[0.45rem] text-faint">
              <LayersIcon className="h-[0.9rem] w-[0.9rem]" />
              <span className="font-mono text-[0.62rem] uppercase tracking-[0.18em]">Sources</span>
            </span>
            {(latestAnswer?.citations.length ?? 0) > 0 && (
              <span className="rounded-full border border-line bg-panel px-[0.45rem] py-[0.05rem]
                               font-mono text-[0.6rem] text-graphite">
                {latestAnswer!.citations.length}
              </span>
            )}
          </div>
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
