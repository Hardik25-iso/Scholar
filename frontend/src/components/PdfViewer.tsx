import { useEffect } from "react";
import { paperFileUrl } from "../api";

export interface ViewerTarget {
  paperDbId: number; // Paper.id (DB row) — what the file endpoint keys on
  title: string;     // human title for the header
  page: number;      // 0-indexed unit to open at
  locator: string;   // "page 12" / "slide 3" — the backend's honest label
  unit: string;      // only "page" (a real PDF) can render in the iframe
  passage: string;   // the cited text, shown when the format cannot be rendered
}

interface Props {
  target: ViewerTarget;
  onClose: () => void;
}

/**
 * Slide-in drawer for verifying a claim against its source.
 *
 * For a PDF this is the browser's native viewer in an <iframe> with a "#page=N"
 * fragment (1-indexed there, 0-indexed here). No other supported format has an
 * in-browser renderer — pointing an iframe at a .docx makes the browser download
 * it instead, which is a worse outcome than saying so. Those show the cited
 * passage verbatim with a download link, which is what verification needs
 * anyway: the exact text, and the original file to check it against.
 */
export default function PdfViewer({ target, onClose }: Props) {
  // Close on Escape — expected for an overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const fileUrl = paperFileUrl(target.paperDbId);
  const renderable = target.unit === "page"; // i.e. a real PDF
  const src = `${fileUrl}#page=${target.page + 1}`;

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
              source · {target.locator}
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
        {renderable ? (
          /* key on src so switching sources reloads the iframe at the new page */
          <iframe key={src} src={src} title={target.title} className="min-h-0 flex-1 bg-white" />
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto bg-white px-6 py-5">
            <p className="mb-3 font-mono text-[0.62rem] uppercase tracking-[0.14em] text-faint">
              cited passage · {target.locator}
            </p>
            <p className="whitespace-pre-wrap font-serif text-[0.9rem] leading-[1.7] text-ink">
              {target.passage}
            </p>
            <p className="mt-6 border-t border-line pt-4 font-mono text-[0.68rem] leading-relaxed text-graphite">
              This format has no in-browser viewer, so the passage above is shown verbatim
              instead of a rendering of the file.{" "}
              <a href={fileUrl} className="text-accent underline underline-offset-2">
                Download the original
              </a>{" "}
              to check it in context.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
