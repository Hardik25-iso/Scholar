import type { Answer } from "../api";

interface Props {
  answer: Answer;
  activeCitation: number | null; // 1-based citation number, or null
  onCite: (n: number | null) => void;
  streaming?: boolean;           // show a live caret while tokens arrive
}

/**
 * Renders the answer text, turning every [n] marker the LLM produced into an
 * interactive superscript that highlights the matching source card.
 */
export default function AnswerView({ answer, activeCitation, onCite, streaming }: Props) {
  // Split on [n] markers, keeping them: "text [2] more" -> ["text ", "[2]", " more"]
  const parts = answer.answer.split(/(\[\d+\])/g);

  return (
    <div className="rise">
      <p className="font-serif text-[1.2rem] leading-[1.85] text-ink">
        {parts.map((part, i) => {
          const m = part.match(/^\[(\d+)\]$/);
          if (!m) return <span key={i}>{part}</span>;

          const n = Number(m[1]);
          // Markers pointing past the source list (LLM slip) render as plain text.
          if (n < 1 || n > answer.citations.length) return <span key={i}>{part}</span>;

          const active = activeCitation === n;
          return (
            <sup key={i}>
              <button
                onClick={() => onCite(active ? null : n)}
                onMouseEnter={() => onCite(n)}
                className={`mx-0.5 cursor-pointer rounded-sm px-1 font-mono text-[0.66em] font-medium
                            transition-colors duration-200 ${
                              active
                                ? "bg-accent text-paper"
                                : "bg-accentSoft text-accent hover:bg-accent hover:text-paper"
                            }`}
              >
                {n}
              </button>
            </sup>
          );
        })}
        {/* live caret while streaming (also covers the gap before the first token) */}
        {streaming && (
          <span className="ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[0.15em] animate-pulse bg-accent" />
        )}
      </p>
    </div>
  );
}
