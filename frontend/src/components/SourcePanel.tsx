import { useEffect, useRef } from "react";
import type { Citation } from "../api";

interface Props {
  citations: Citation[];
  activeCitation: number | null; // 1-based
  onCite: (n: number | null) => void;
  onOpen?: (citation: Citation) => void; // open this source in the PDF viewer
}

/** Relevance drawn as a thin tick bar — rerank logits roughly span -4..+8. */
function ScoreBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, ((value + 4) / 12) * 100));
  return (
    <span className="relative inline-block h-[3px] w-16 rounded-full bg-line align-middle">
      <span
        className="absolute inset-y-0 left-0 rounded-full bg-accent/70"
        style={{ width: `${pct}%` }}
      />
    </span>
  );
}

/** Right pane: the exact passages behind the current answer. */
export default function SourcePanel({ citations, activeCitation, onCite, onOpen }: Props) {
  const refs = useRef<(HTMLDivElement | null)[]>([]);

  // When a citation marker is clicked in the answer, scroll its card into view.
  useEffect(() => {
    if (activeCitation !== null) {
      refs.current[activeCitation - 1]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeCitation]);

  if (citations.length === 0) {
    return (
      <p className="px-6 pt-8 font-mono text-xs leading-relaxed text-faint">
        Passages retrieved for your question will appear here — every answer is
        grounded in, and cited to, these exact excerpts.
      </p>
    );
  }

  return (
    <div className="space-y-3 px-5 pb-8">
      {citations.map((c, i) => {
        const n = i + 1;
        const active = activeCitation === n;
        return (
          <div
            key={n}
            ref={(el) => (refs.current[i] = el)}
            onMouseEnter={() => onCite(n)}
            onMouseLeave={() => onCite(null)}
            onClick={() => onOpen?.(c)}
            className={`rise border bg-panel p-4 transition-colors duration-200 ${
              onOpen ? "cursor-pointer" : "cursor-default"
            } ${active ? "border-accent/60" : "border-line"}`}
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className="mb-2 flex items-baseline justify-between gap-2">
              <span
                className={`font-mono text-xs font-medium ${
                  active ? "text-accent" : "text-graphite"
                }`}
              >
                [{n}] {c.paper_id.replace(/_/g, " ")}
              </span>
              <span className="shrink-0 font-mono text-[0.65rem] text-faint">
                p.{c.page + 1}{onOpen && <span className="ml-1 text-accent">↗</span>}
              </span>
            </div>
            <p className="mb-3 line-clamp-5 text-[0.8rem] leading-relaxed text-graphite">
              {c.text}
            </p>
            <div className="flex items-center gap-2 font-mono text-[0.62rem] text-faint">
              {c.rerank_score !== null && (
                <>
                  <ScoreBar value={c.rerank_score} />
                  <span>rerank {c.rerank_score.toFixed(2)}</span>
                  <span aria-hidden>·</span>
                </>
              )}
              <span>cos {c.score.toFixed(3)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
