import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ApiError, resetPassword } from "../api";
import { useAuth } from "../auth/AuthContext";
import AuthShell, { ERROR, FIELD, LABEL, SUBMIT } from "../components/AuthShell";

/**
 * Choose a new password, using the single-use token from the emailed link.
 *
 * The token stays in the URL and is never stored: this page is reached once,
 * consumed once, and the backend burns every outstanding token for the account
 * on success.
 */
export default function ResetPassword() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const { adopt } = useAuth();
  const navigate = useNavigate();

  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      // The route sets fresh auth cookies, so the user is already signed in by
      // the time this resolves — adopt the session rather than bounce to login.
      adopt(await resetPassword(token, password));
      navigate("/app", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  // A link that arrived without its token cannot be repaired here, and a form
  // that would fail on submit is worse than saying so up front.
  if (!token)
    return (
      <AuthShell label="reset password" heading={"That link is\nincomplete."}>
        <p className="mt-6 text-sm leading-relaxed text-graphite">
          The reset link is missing its token. Some mail clients break long links
          across lines — request a fresh one and open it in a single click.
        </p>
        <p className="mt-6 font-sans text-sm text-graphite">
          <Link to="/forgot" className="text-accent underline-offset-2 hover:underline">
            Request a new link
          </Link>
        </p>
      </AuthShell>
    );

  return (
    <AuthShell label="reset password" heading={"Choose a new\npassword."}>
      <form onSubmit={submit} className="mt-8 space-y-4">
        <div>
          <label htmlFor="password" className={LABEL}>new password</label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={FIELD}
          />
          <p className="mt-1.5 font-mono text-[0.6rem] text-faint">at least 8 characters</p>
        </div>

        {error && <p role="alert" className={ERROR}>{error}</p>}

        <button type="submit" disabled={busy} className={SUBMIT}>
          {busy ? "…" : "Set password and sign in"}
        </button>
      </form>

      <p className="mt-6 font-sans text-sm text-graphite">
        Link expired?{" "}
        <Link to="/forgot" className="text-accent underline-offset-2 hover:underline">
          Request a new one
        </Link>
      </p>
    </AuthShell>
  );
}
