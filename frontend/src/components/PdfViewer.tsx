import { useEffect } from "react";
import { paperFileUrl } from "../api";

export interface ViewerTarget {
  paperDbId: number; // Paper.id (DB row) — what the file endpoint keys on
  title: string;     // human title for the header
  page: number;      // 0-indexed page to open at
}

interface Props {
  target: ViewerTarget;
  onClose: () => void;
}

/**
 * Slide-in drawer that renders the source PDF at the cited page, so a claim can
 * be verified against the original. Uses the browser's native PDF viewer via an
 * <iframe> with a "#page=N" fragment (pages are 1-indexed there, ours are 0).
 */
export default function PdfViewer({ target, onClose }: Props) {
  // Close on Escape — expected for an overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const src = `${paperFileUrl(target.paperDbId)}#page=${target.page + 1}`;

  return (
    <div className="fixed inset-0 z-40 flex">
      {/* backdrop */}
      <div className="flex-1 bg-ink/30 backdrop-blur-[1px]" onClick={onClose} />
      {/* drawer */}
      <div className="flex h-full w-full max-w-3xl flex-col border-l border-line bg-panel shadow-2xl">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <div className="min-w-0">
            <p className="truncate font-serif text-sm font-medium text-ink" title={target.title}>
              {target.title.replace(/_/g, " ")}
            </p>
            <p className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-faint">
              source · page {target.page + 1}
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 shrink-0 cursor-pointer font-mono text-[0.7rem] uppercase tracking-[0.14em]
                       text-graphite transition-colors hover:text-accent"
          >
            close ✕
          </button>
        </div>
        {/* key on src so switching sources reloads the iframe at the new page */}
        <iframe key={src} src={src} title={target.title} className="min-h-0 flex-1 bg-white" />
      </div>
    </div>
  );
}
