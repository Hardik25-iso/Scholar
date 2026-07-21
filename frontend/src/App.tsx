import { useEffect, useRef, useState } from "react";
import { ask, type Answer } from "./api";
import AnswerView from "./components/AnswerView";
import QueryBar from "./components/QueryBar";
import SourcePanel from "./components/SourcePanel";

interface Exchange {
  question: string;
  answer: Answer | null; // null while the pipeline is running
  error: string | null;
}

const SUGGESTIONS = [
  "How does multi-head attention work?",
  "Why is the dot product scaled by 1/√dk?",
  "How does the Transformer handle word order without recurrence?",
];

export default function App() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [busy, setBusy] = useState(false);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const latestAnswer = [...exchanges].reverse().find((e) => e.answer)?.answer;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [exchanges]);

  const handleAsk = async (question: string) => {
    setBusy(true);
    setActiveCitation(null);
    setExchanges((xs) => [...xs, { question, answer: null, error: null }]);
    try {
      const answer = await ask(question);
      setExchanges((xs) =>
        xs.map((x, i) => (i === xs.length - 1 ? { ...x, answer } : x)),
      );
    } catch (err) {
      setExchanges((xs) =>
        xs.map((x, i) =>
          i === xs.length - 1
            ? { ...x, error: err instanceof Error ? err.message : String(err) }
            : x,
        ),
      );
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
        <span className="font-mono text-[0.62rem] text-faint">
          gemma3:4b · local · attention_is_all_you_need
        </span>
      </header>

      {/* ——— two-pane workspace ——— */}
      <div className="flex min-h-0 flex-1">
        {/* chat pane */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-6">
            <div className="mx-auto max-w-2xl py-10">
              {exchanges.length === 0 && (
                <div className="rise mt-[18vh]">
                  <p className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
                    01 — ask
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
                      />
                    )}
                    {x.error && (
                      <p className="font-mono text-xs leading-relaxed text-accent">
                        {x.error} — is the backend running on :8000 and Ollama up?
                      </p>
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
          <QueryBar disabled={busy} onAsk={handleAsk} />
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
          />
        </aside>
      </div>
    </div>
  );
}
