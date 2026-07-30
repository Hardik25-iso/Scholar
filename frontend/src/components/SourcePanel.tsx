import { useEffect, useRef } from "react";
import type { Citation } from "../api";
import { ExternalIcon } from "./icons";

interface Props {
  citations: Citation[];
  activeCitation: number | null; // 1-based
  onCite: (n: number | null) => void;
  onOpen?: (citation: Citation) => void; // open this source in the PDF viewer
}

/** Relevance drawn as a filled meter — rerank logits roughly span -4..+8. */
function Meter({ value }: { value: number }) {
  const pct = Math.max(4, Math.min(100, ((value + 4) / 12) * 100));
  return (
    <span className="relative h-1 flex-1 overflow-hidden rounded-sm bg-line">
      <span
        className="absolute inset-y-0 left-0 rounded-sm bg-gradient-to-r from-[#C7614F] to-accent"
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
      refs.current[activeCitation - 1]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeCitation]);

  if (citations.length === 0) {
    return (
      <p className="px-5 pt-8 font-mono text-[0.7rem] leading-relaxed text-faint">
        Passages retrieved for your question will appear here — every answer is
        grounded in, and cited to, these exact excerpts.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-[0.6rem] px-[0.9rem] pb-8 pt-[0.85rem]">
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
            className={`rise rounded-[11px] border bg-raised p-[0.85rem] shadow-e1 transition-all duration-200
                        hover:-translate-y-0.5 hover:shadow-e2 ${onOpen ? "cursor-pointer" : "cursor-default"} ${
                          active ? "border-accent/60 shadow-e2 ring-4 ring-accent/[0.08]" : "border-lineSoft"
                        }`}
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className="mb-2 flex items-center gap-[0.45rem]">
              <span className="grid h-[1.15rem] w-[1.15rem] shrink-0 place-items-center rounded-[5px]
                               bg-accent font-mono text-[0.62rem] font-semibold text-paper">
                {n}
              </span>
              <span className="min-w-0 flex-1 truncate font-mono text-[0.66rem] text-graphite">
                {c.paper_id.replace(/_/g, " ")}
              </span>
              <span className="inline-flex shrink-0 items-center gap-1 font-mono text-[0.62rem] text-faint">
                p.{c.page + 1}
                {onOpen && <ExternalIcon className="h-[0.7rem] w-[0.7rem] text-accent" />}
              </span>
            </div>
            <p className="mb-[0.6rem] line-clamp-3 font-serif text-[0.82rem] leading-[1.55] text-graphite">
              {c.text}
            </p>
            <div className="flex items-center gap-2 font-mono text-[0.62rem] text-faint">
              {c.rerank_score !== null ? (
                <>
                  <span className="shrink-0">rerank {c.rerank_score >= 0 ? "+" : ""}{c.rerank_score.toFixed(2)}</span>
                  <Meter value={c.rerank_score} />
                </>
              ) : (
                <span>cos {c.score.toFixed(3)}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
