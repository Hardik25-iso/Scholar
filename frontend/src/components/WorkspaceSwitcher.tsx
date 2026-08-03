import { useEffect, useRef, useState } from "react";
import {
  acceptInvitation,
  activateWorkspace,
  createWorkspace,
  inviteMember,
  listMembers,
  listWorkspaces,
  type Member,
  type Workspace,
} from "../api";
import { CheckIcon, ChevronDownIcon, PlusIcon, UsersIcon } from "./icons";

interface Props {
  /** Called after the active workspace changes, so the library reloads. */
  onSwitch: () => void;
}

/**
 * Which library am I looking at, and who else can see it.
 *
 * Everything below the switcher exists because a shared library is only safe if
 * the answer to "who else can read this" is visible without going looking for
 * it. Membership is shown inline rather than behind a settings page.
 */
export default function WorkspaceSwitcher({ onSwitch }: Props) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [open, setOpen] = useState(false);
  const [members, setMembers] = useState<Member[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const current = workspaces.find((w) => w.is_current) ?? null;

  const refresh = async () => {
    try {
      const list = await listWorkspaces();
      setWorkspaces(list);
      const active = list.find((w) => w.is_current);
      setMembers(active && !active.is_personal ? await listMembers(active.id) : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  // Close on an outside click or Escape — expected of a dropdown, and without
  // it the panel sits over the library and swallows clicks meant for it.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await action();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const switchTo = (w: Workspace) =>
    run(async () => {
      if (w.is_current) return;
      await activateWorkspace(w.id);
      setOpen(false);
      onSwitch(); // the library and the conversation belong to the old workspace
    });

  const create = () => {
    const name = window.prompt("Name the workspace");
    if (name?.trim()) run(() => createWorkspace(name.trim()));
  };

  const invite = () => {
    if (!current) return;
    const email = window.prompt(`Invite someone to "${current.name}" by email`);
    if (!email?.trim()) return;
    run(async () => {
      const invitation = await inviteMember(current.id, email.trim());
      // The token is shown because invitation email is not wired up yet — the
      // backend says so too. Surfacing it is a stopgap, and labelling it as one
      // is the difference between a known gap and a mysterious string.
      setNotice(
        `No email is sent yet — send ${invitation.email} this invite code:\n\n${invitation.token}`,
      );
    });
  };

  const join = () => {
    const token = window.prompt("Paste an invite code");
    if (token?.trim()) run(() => acceptInvitation(token.trim()));
  };

  if (!current) return null;

  return (
    <div ref={panelRef} className="relative border-b border-line px-3 py-2">
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={busy}
        className="flex w-full items-center gap-2 rounded-[8px] px-2 py-1.5 text-left
                   transition-colors hover:bg-panel disabled:opacity-60"
      >
        <UsersIcon className="h-[0.9rem] w-[0.9rem] shrink-0 text-accent" />
        <span className="min-w-0 flex-1">
          <span className="block truncate font-sans text-[0.78rem] font-semibold text-ink">
            {current.name}
          </span>
          <span className="block font-mono text-[0.58rem] uppercase tracking-[0.14em] text-faint">
            {current.is_personal ? "personal" : `${members.length} member${members.length === 1 ? "" : "s"}`}
          </span>
        </span>
        <ChevronDownIcon className="h-[0.8rem] w-[0.8rem] shrink-0 text-faint" />
      </button>

      {open && (
        <div className="absolute left-3 right-3 top-full z-20 mt-1 overflow-hidden rounded-[10px]
                        border border-line bg-paper shadow-e2">
          <ul className="max-h-60 overflow-y-auto py-1">
            {workspaces.map((w) => (
              <li key={w.id}>
                <button
                  onClick={() => switchTo(w)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left
                             transition-colors hover:bg-panel"
                >
                  <span className="w-[0.9rem] shrink-0">
                    {w.is_current && <CheckIcon className="h-[0.8rem] w-[0.8rem] text-accent" />}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-sans text-[0.76rem] text-ink">
                    {w.name}
                  </span>
                  <span className="shrink-0 font-mono text-[0.55rem] uppercase tracking-[0.12em] text-faint">
                    {w.is_personal ? "personal" : w.role}
                  </span>
                </button>
              </li>
            ))}
          </ul>

          <div className="border-t border-line">
            {[
              { label: "New workspace", action: create },
              { label: "Join with a code", action: join },
              // A personal library is deliberately unshareable — the backend
              // refuses, so the option should not be offered.
              ...(current.is_personal || current.role !== "owner"
                ? []
                : [{ label: "Invite someone", action: invite }]),
            ].map(({ label, action }) => (
              <button
                key={label}
                onClick={action}
                disabled={busy}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left font-mono
                           text-[0.66rem] text-graphite transition-colors hover:bg-panel
                           hover:text-accent disabled:opacity-60"
              >
                <PlusIcon className="h-[0.7rem] w-[0.7rem]" />
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Who else can read this library. Shown inline, not behind a settings
          page: it is the question a shared library most needs to answer. */}
      {!current.is_personal && members.length > 0 && (
        <ul className="mt-1 flex flex-wrap gap-1 px-2 pb-1">
          {members.map((m) => (
            <li
              key={m.user_id}
              title={`${m.email} — ${m.role}`}
              className="max-w-full truncate rounded-full border border-lineSoft bg-panel
                         px-[0.4rem] py-[0.05rem] font-mono text-[0.55rem] text-graphite"
            >
              {m.email.split("@")[0]}
              {m.role === "owner" && <span className="text-accent"> ★</span>}
            </li>
          ))}
        </ul>
      )}

      {error && <p className="px-2 pb-1 font-mono text-[0.6rem] text-accent">{error}</p>}
      {notice && (
        <p className="mx-2 mb-1 whitespace-pre-wrap break-all rounded-[6px] border border-lineSoft
                      bg-panel p-2 font-mono text-[0.58rem] leading-relaxed text-graphite">
          {notice}
        </p>
      )}
    </div>
  );
}
