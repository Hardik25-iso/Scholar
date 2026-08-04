import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { acceptInvitation, activateWorkspace, ApiError } from "../api";
import { useAuth } from "../auth/AuthContext";
import AuthShell, { ERROR, SUBMIT } from "../components/AuthShell";

/**
 * Accept a workspace invitation from an emailed link.
 *
 * Deliberately NOT automatic on mount. Joining a workspace means someone else's
 * documents become readable to this account and this account becomes visible to
 * everyone in it — a consequence that should follow a click, not a page load.
 * It also makes the "you are signed in as the wrong account" case survivable:
 * the address is on screen before anything happens.
 */
export default function JoinWorkspace() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const { user } = useAuth();
  const navigate = useNavigate();

  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const accept = async () => {
    setError(null);
    setBusy(true);
    try {
      const workspace = await acceptInvitation(token);
      // Land them IN the library they just joined. Accepting and then showing
      // their own empty personal library would read as though nothing happened.
      await activateWorkspace(workspace.id);
      navigate("/app", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  if (!token)
    return (
      <AuthShell label="invitation" heading={"That link is\nincomplete."}>
        <p className="mt-6 text-sm leading-relaxed text-graphite">
          The invitation link is missing its code. Ask whoever invited you to send
          it again.
        </p>
        <p className="mt-6 font-sans text-sm text-graphite">
          <Link to="/app" className="text-accent underline-offset-2 hover:underline">
            Go to your library
          </Link>
        </p>
      </AuthShell>
    );

  return (
    <AuthShell label="invitation" heading={"You have been\ninvited."}>
      <p className="mt-6 text-sm leading-relaxed text-graphite">
        Accepting adds this account —{" "}
        <span className="font-mono text-[0.8rem] text-ink">{user?.email}</span> — to
        the workspace. You will be able to read and question its documents, and
        its members will see that you joined.
      </p>
      <p className="mt-3 font-mono text-[0.62rem] leading-relaxed text-faint">
        An invitation is issued to one address. If this is not the account it was
        sent to, sign out and sign in as that one first.
      </p>

      {error && <p role="alert" className={`mt-4 ${ERROR}`}>{error}</p>}

      <button onClick={accept} disabled={busy} className={`mt-6 ${SUBMIT}`}>
        {busy ? "…" : "Accept invitation"}
      </button>

      <p className="mt-6 font-sans text-sm text-graphite">
        <Link to="/app" className="text-accent underline-offset-2 hover:underline">
          Not now
        </Link>
      </p>
    </AuthShell>
  );
}
