import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

/**
 * Public landing page — Scholar's front door. Sells the core idea: answers
 * grounded in YOUR papers, every claim cited to an exact page, not a chatbot
 * guessing from memory. Keeps the manuscript palette but at a bolder, more
 * editorial scale than the working app.
 */
export default function Landing() {
  const { user } = useAuth();

  // CTA target depends on whether you're already signed in.
  const primary = user
    ? { to: "/app", label: "Open workspace" }
    : { to: "/signup", label: "Get started — it's free" };

  return (
    <div className="min-h-screen bg-paper text-ink">
      {/* ——— nav ——— */}
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <div className="flex items-baseline gap-2">
          <span className="font-serif text-xl font-semibold tracking-tight">Scholar</span>
          <span className="hidden font-mono text-[0.6rem] uppercase tracking-[0.18em] text-faint sm:inline">
            grounded research
          </span>
        </div>
        <div className="flex items-center gap-5 font-mono text-[0.72rem] uppercase tracking-[0.12em]">
          {user ? (
            <Link to="/app" className="text-graphite transition-colors hover:text-accent">
              open workspace →
            </Link>
          ) : (
            <>
              <Link to="/login" className="text-graphite transition-colors hover:text-accent">
                sign in
              </Link>
              <Link
                to="/signup"
                className="border border-ink px-4 py-2 text-ink transition-colors hover:bg-ink hover:text-paper"
              >
                get started
              </Link>
            </>
          )}
        </div>
      </nav>

      {/* ——— hero ——— */}
      <header className="mx-auto grid max-w-6xl gap-12 px-6 pb-24 pt-16 md:grid-cols-[1.1fr_0.9fr] md:pt-24">
        <div className="rise">
          <p className="font-mono text-[0.62rem] uppercase tracking-[0.2em] text-faint">
            retrieval, not recall
          </p>
          <h1 className="mt-4 font-serif text-5xl font-medium leading-[1.05] tracking-tight md:text-6xl">
            Ask the papers,
            <br />
            <em className="text-accent">not the model.</em>
          </h1>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-graphite">
            Upload your PDFs and ask questions in plain language. Every answer is
            assembled only from passages retrieved out of your own papers — and
            cites the exact page each claim came from.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link
              to={primary.to}
              className="bg-accent px-6 py-3 font-mono text-[0.75rem] uppercase tracking-[0.12em]
                         text-paper transition-opacity hover:opacity-90"
            >
              {primary.label}
            </Link>
            <Link
              to={user ? "/app" : "/login"}
              className="font-mono text-[0.75rem] uppercase tracking-[0.12em] text-graphite
                         underline-offset-4 transition-colors hover:text-accent hover:underline"
            >
              {user ? "go to your library" : "sign in"}
            </Link>
          </div>
          <p className="mt-6 font-mono text-[0.62rem] text-faint">
            Runs a local model — your papers never leave your machine.
          </p>
        </div>

        {/* mock answer card — shows the product's payoff at a glance */}
        <div className="rise flex items-center" style={{ animationDelay: "120ms" }}>
          <AnswerMock />
        </div>
      </header>

      {/* ——— the difference ——— */}
      <section className="border-y border-line bg-panel/50">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <p className="font-mono text-[0.62rem] uppercase tracking-[0.2em] text-faint">
            01 — why it's different
          </p>
          <h2 className="mt-3 max-w-2xl font-serif text-3xl font-medium leading-snug">
            A chatbot recites from memory. Scholar answers from evidence.
          </h2>
          <div className="mt-10 grid gap-8 md:grid-cols-3">
            <Feature
              title="Grounded, not guessed"
              body="Answers are built only from passages retrieved out of your papers. If the papers don't say it, Scholar says so — instead of inventing."
            />
            <Feature
              title="Cited to the page"
              body="Every claim carries a [n] marker linking to the exact source passage and page. Click it, read it, trust it."
            />
            <Feature
              title="Your library, private"
              body="Sign in, upload your own PDFs, and ask across them. Generation runs locally — nothing is sent to a third-party model."
            />
          </div>
        </div>
      </section>

      {/* ——— how it works ——— */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <p className="font-mono text-[0.62rem] uppercase tracking-[0.2em] text-faint">
          02 — how it works
        </p>
        <div className="mt-10 grid gap-10 md:grid-cols-3">
          <Step n="1" title="Upload your papers" body="Drop in PDFs — research papers, reports, textbooks. Scholar parses and indexes them into your private library." />
          <Step n="2" title="Ask in plain language" body="Type a question the way you'd ask a colleague. Scholar retrieves the most relevant passages and reranks them." />
          <Step n="3" title="Read cited answers" body="Get a concise answer that streams in live, with every claim linked to the exact page it came from." />
        </div>
      </section>

      {/* ——— closing CTA ——— */}
      <section className="border-t border-line">
        <div className="mx-auto max-w-6xl px-6 py-20 text-center">
          <h2 className="font-serif text-4xl font-medium leading-tight">
            Stop trusting. <em className="text-accent">Start verifying.</em>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-graphite">
            Turn a stack of papers into answers you can actually check.
          </p>
          <Link
            to={primary.to}
            className="mt-8 inline-block bg-ink px-8 py-3 font-mono text-[0.75rem] uppercase
                       tracking-[0.12em] text-paper transition-opacity hover:opacity-90"
          >
            {primary.label}
          </Link>
        </div>
      </section>

      {/* ——— footer ——— */}
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-6 py-6
                        font-mono text-[0.62rem] text-faint sm:flex-row">
          <span>Scholar — grounded answers from your papers</span>
          <span>retrieval · rerank · cite</span>
        </div>
      </footer>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="border-l-2 border-accent/40 pl-4">
      <h3 className="font-serif text-lg font-medium text-ink">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-graphite">{body}</p>
    </div>
  );
}

function Step({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div>
      <span className="font-mono text-2xl font-medium text-accent">{n}</span>
      <h3 className="mt-3 font-serif text-lg font-medium text-ink">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-graphite">{body}</p>
    </div>
  );
}

/** A stylised, static preview of a cited answer — the product in one glance. */
function AnswerMock() {
  return (
    <div className="w-full border border-line bg-panel p-6 shadow-[0_1px_0_rgba(0,0,0,0.02)]">
      <p className="mb-3 border-l-2 border-accent pl-3 font-mono text-[0.7rem] text-graphite">
        How does multi-head attention work?
      </p>
      <p className="font-serif text-[1.05rem] leading-[1.8] text-ink">
        Multi-head attention runs several attention layers in parallel
        <Sup n="1" />, each projecting queries, keys and values into a lower
        dimension with learned projections
        <Sup n="2" />, then concatenating the results.
      </p>
      <div className="mt-5 space-y-2 border-t border-line pt-4">
        <SourceChip n="1" page="p.4" />
        <SourceChip n="2" page="p.5" />
      </div>
    </div>
  );
}

function Sup({ n }: { n: string }) {
  return (
    <sup>
      <span className="mx-0.5 rounded-sm bg-accentSoft px-1 font-mono text-[0.66em] font-medium text-accent">
        {n}
      </span>
    </sup>
  );
}

function SourceChip({ n, page }: { n: string; page: string }) {
  return (
    <div className="flex items-baseline gap-2 font-mono text-[0.62rem] text-faint">
      <span className="text-accent">[{n}]</span>
      <span className="text-graphite">attention is all you need</span>
      <span>· {page}</span>
    </div>
  );
}
