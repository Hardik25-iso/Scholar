import { useRef } from "react";
import type { Paper } from "../api";
import { FileIcon, TrashIcon, UploadIcon } from "./icons";
import WorkspaceSwitcher from "./WorkspaceSwitcher";

interface Props {
  papers: Paper[];
  loading: boolean;      // initial library fetch
  uploading: boolean;    // an upload is in flight
  error: string | null;
  onUpload: (file: File) => void;
  onDelete: (id: number) => void;
  onSwitchWorkspace: () => void;
}

/**
 * Left rail: the documents the current WORKSPACE holds. Every answer is
 * grounded in exactly these, so the switcher sits directly above them — which
 * library you are asking is not a settings-page detail, it is the single most
 * important piece of context on the screen.
 */
export default function Library({
  papers, loading, uploading, error, onUpload, onDelete, onSwitchWorkspace,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const pick = () => fileRef.current?.click();
  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onUpload(f);
    e.target.value = ""; // allow re-selecting the same file later
  };

  return (
    <aside className="hidden w-[264px] shrink-0 flex-col overflow-y-auto border-r border-line
                      bg-gradient-to-b from-panel/60 to-paper/20 md:flex">
      <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line
                      bg-paper px-[1.15rem] py-[0.85rem]">
        <span className="flex items-center gap-[0.45rem] text-faint">
          <FileIcon className="h-[0.9rem] w-[0.9rem]" />
          <span className="font-mono text-[0.62rem] uppercase tracking-[0.18em]">Library</span>
        </span>
        <span className="rounded-full border border-line bg-panel px-[0.45rem] py-[0.05rem]
                         font-mono text-[0.6rem] text-graphite">
          {papers.length}
        </span>
      </div>

      <WorkspaceSwitcher onSwitch={onSwitchWorkspace} />

      {/* upload dropzone */}
      {/* Keep in sync with parser.SUPPORTED_EXTENSIONS — the backend gates on the
          extension, so the picker should offer exactly what it will accept. */}
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.docx,.pptx,.xlsx,.txt,.md"
        hidden
        onChange={onFile}
      />
      <button
        onClick={pick}
        disabled={uploading}
        className="group mx-4 mt-[0.9rem] cursor-pointer rounded-[10px] border-[1.5px] border-dashed
                   border-line bg-panel px-4 py-5 text-center text-graphite transition-all duration-200
                   hover:border-accent/50 hover:bg-[#FDF6F4] hover:text-accentInk hover:shadow-e1
                   disabled:cursor-wait disabled:opacity-70"
      >
        <UploadIcon className="mx-auto mb-2 h-[1.6rem] w-[1.6rem] text-accent" />
        <b className="block font-sans text-[0.82rem] font-semibold">Drop a document to add it</b>
        <span className="mt-0.5 block font-mono text-[0.6rem] tracking-[0.04em] text-faint">
          pdf · docx · pptx · xlsx · txt · md — max 20 MB
        </span>
      </button>
      {error && <p className="mx-4 mt-2 font-mono text-[0.62rem] leading-relaxed text-accent">{error}</p>}

      {/* paper cards */}
      <div className="flex min-h-0 flex-1 flex-col gap-[0.55rem] px-[0.8rem] pb-4 pt-3">
        {uploading && (
          <div className="flex gap-[0.7rem] rounded-[10px] border border-lineSoft bg-raised p-[0.7rem] px-3 shadow-e1">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[7px] bg-accentSoft text-accent">
              <FileIcon className="h-[1.05rem] w-[1.05rem]" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="font-mono text-[0.7rem] text-accent">indexing…</p>
              <div className="mt-[0.4rem] h-[3px] overflow-hidden rounded-sm bg-line">
                <span className="block h-full w-3/5 animate-pulse rounded-sm bg-gradient-to-r from-accent to-[#C7614F]" />
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <p className="px-3 py-3 font-mono text-[0.62rem] text-faint">loading…</p>
        ) : papers.length === 0 && !uploading ? (
          <p className="px-3 py-3 font-mono text-[0.62rem] leading-relaxed text-faint">
            No documents yet. Upload one to start asking grounded questions.
          </p>
        ) : (
          papers.map((p, i) => (
            <div
              key={p.id}
              className="rise group relative flex cursor-pointer gap-[0.7rem] rounded-[10px] border
                         border-lineSoft bg-raised p-[0.7rem] px-3 shadow-e1 transition-all duration-200
                         hover:-translate-y-0.5 hover:border-line hover:shadow-e2"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[7px] bg-accentSoft text-accent">
                <FileIcon className="h-[1.05rem] w-[1.05rem]" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate font-serif text-[0.94rem] leading-tight text-ink" title={p.title}>
                  {p.title.replace(/_/g, " ")}
                </p>
                <p className="mt-1 font-mono text-[0.6rem] text-faint">{p.n_chunks} chunks</p>
              </div>
              <button
                onClick={() => onDelete(p.id)}
                title="Remove paper"
                aria-label={`Remove ${p.title}`}
                className="absolute right-2 top-2 cursor-pointer text-faint opacity-0
                           transition-opacity duration-150 hover:text-accent group-hover:opacity-100"
              >
                <TrashIcon className="h-[0.85rem] w-[0.85rem]" />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
