import { useRef } from "react";
import type { Paper } from "../api";

interface Props {
  papers: Paper[];
  loading: boolean;      // initial library fetch
  uploading: boolean;    // an upload is in flight
  error: string | null;
  onUpload: (file: File) => void;
  onDelete: (id: number) => void;
}

/**
 * Left rail: the user's paper library. Upload adds a PDF (parsed + embedded
 * into their private index); every answer is grounded in exactly these papers.
 */
export default function Library({ papers, loading, uploading, error, onUpload, onDelete }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const pick = () => fileRef.current?.click();
  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onUpload(f);
    e.target.value = ""; // allow re-selecting the same file later
  };

  return (
    <aside className="hidden w-[17rem] shrink-0 flex-col border-r border-line bg-panel/40 md:flex">
      <div className="flex items-baseline justify-between border-b border-line px-5 py-3">
        <span className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
          library
        </span>
        <span className="font-mono text-[0.62rem] text-faint">{papers.length}</span>
      </div>

      {/* upload */}
      <div className="border-b border-line p-4">
        <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={onFile} />
        <button
          onClick={pick}
          disabled={uploading}
          className="w-full cursor-pointer border border-dashed border-line px-3 py-4 text-center
                     font-mono text-[0.7rem] uppercase tracking-[0.14em] text-graphite
                     transition-colors duration-200 hover:border-accent/50 hover:text-accent
                     disabled:cursor-wait disabled:opacity-60"
        >
          {uploading ? "indexing paper…" : "+ upload pdf"}
        </button>
        {error && <p className="mt-2 font-mono text-[0.62rem] leading-relaxed text-accent">{error}</p>}
      </div>

      {/* list */}
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {loading ? (
          <p className="px-3 py-3 font-mono text-[0.62rem] text-faint">loading…</p>
        ) : papers.length === 0 ? (
          <p className="px-3 py-3 font-mono text-[0.62rem] leading-relaxed text-faint">
            No papers yet. Upload a PDF to start asking grounded questions.
          </p>
        ) : (
          papers.map((p) => (
            <div
              key={p.id}
              className="group flex items-start justify-between gap-2 rounded px-3 py-2
                         transition-colors duration-150 hover:bg-paper"
            >
              <div className="min-w-0">
                <p className="truncate font-serif text-[0.9rem] leading-snug text-ink" title={p.title}>
                  {p.title.replace(/_/g, " ")}
                </p>
                <p className="mt-0.5 font-mono text-[0.6rem] text-faint">{p.n_chunks} chunks</p>
              </div>
              <button
                onClick={() => onDelete(p.id)}
                title="Remove paper"
                className="shrink-0 cursor-pointer font-mono text-[0.7rem] text-faint opacity-0
                           transition-opacity duration-150 hover:text-accent group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
