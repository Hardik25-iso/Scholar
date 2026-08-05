import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { accountExportUrl, ApiError, deleteAccount } from "../api";
import { useAuth } from "../auth/AuthContext";
import { ChevronDownIcon } from "./icons";

/**
 * The account menu: who you are signed in as, export, delete, sign out.
 *
 * Export and deletion live here rather than on a settings page nobody visits.
 * A right you cannot find is not much of a right, and both are one click from
 * the address they apply to.
 */
export default function AccountMenu() {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const close = () => {
    setOpen(false);
    setConfirming(false);
    setPassword("");
    setError(null);
  };

  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteAccount(password);
      // The server has cleared the cookies; drop the client's idea of a user so
      // the app leaves the workspace instead of rendering it against a dead
      // session and 401-ing on the next request.
      await logout().catch(() => {});
      window.location.assign("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
      setBusy(false);
    }
  };

  return (
    <div ref={panelRef} className="relative">
      <button
        onClick={() => (open ? close() : setOpen(true))}
        className="inline-flex cursor-pointer items-center gap-[0.35rem] font-mono text-[0.66rem]
                   text-graphite transition-colors duration-200 hover:text-accent"
      >
        <span className="hidden max-w-[14rem] truncate sm:inline">{user?.email}</span>
        <ChevronDownIcon className="h-[0.75rem] w-[0.75rem]" />
      </button>

      {open && (
        <div className="absolute right-0 top-full z-30 mt-2 w-[19rem] overflow-hidden rounded-[10px]
                        border border-line bg-paper shadow-e2">
          {!confirming ? (
            <div className="py-1">
              <a
                href={accountExportUrl()}
                className="block px-4 py-2 font-mono text-[0.68rem] text-graphite
                           transition-colors hover:bg-panel hover:text-accent"
              >
                export my data
                <span className="mt-0.5 block font-sans text-[0.72rem] text-faint">
                  documents, answers and evidence, as a .tar.gz
                </span>
              </a>
              <Link
                to="/privacy"
                onClick={close}
                className="block px-4 py-2 font-mono text-[0.68rem] text-graphite
                           transition-colors hover:bg-panel hover:text-accent"
              >
                how your data is handled
              </Link>
              <button
                onClick={() => logout()}
                className="block w-full px-4 py-2 text-left font-mono text-[0.68rem] text-graphite
                           transition-colors hover:bg-panel hover:text-accent"
              >
                log out
              </button>
              <div className="mt-1 border-t border-line">
                <button
                  onClick={() => setConfirming(true)}
                  className="block w-full px-4 py-2 text-left font-mono text-[0.68rem] text-accent
                             transition-colors hover:bg-[#FDF6F4]"
                >
                  delete my account
                </button>
              </div>
            </div>
          ) : (
            <div className="p-4">
              <p className="font-sans text-[0.82rem] font-semibold text-ink">
                Delete this account?
              </p>
              {/* Says what goes, specifically. "This cannot be undone" without
                  naming what is destroyed is a warning people click through. */}
              <p className="mt-1.5 text-[0.78rem] leading-relaxed text-graphite">
                Your documents, indexes, and every saved answer are removed from the
                server immediately. This cannot be undone. Export first if you want a
                copy.
              </p>
              <label htmlFor="confirm-password" className="mt-3 block font-mono
                                                           text-[0.6rem] uppercase tracking-[0.14em] text-graphite">
                confirm your password
              </label>
              <input
                id="confirm-password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 w-full border border-line bg-panel px-3 py-2 font-sans text-sm
                           text-ink focus:border-accent focus:outline-none"
              />
              {error && (
                <p role="alert" className="mt-2 font-mono text-[0.62rem] leading-relaxed text-accent">
                  {error}
                </p>
              )}
              <div className="mt-3 flex gap-2">
                <button
                  onClick={remove}
                  disabled={busy || !password}
                  className="flex-1 cursor-pointer bg-accent px-3 py-2 font-mono text-[0.62rem]
                             uppercase tracking-[0.12em] text-paper transition-opacity
                             hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? "…" : "delete forever"}
                </button>
                <button
                  onClick={() => { setConfirming(false); setPassword(""); setError(null); }}
                  className="cursor-pointer border border-line px-3 py-2 font-mono text-[0.62rem]
                             uppercase tracking-[0.12em] text-graphite transition-colors
                             hover:text-ink"
                >
                  cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
