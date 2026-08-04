import { useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate, type Location } from "react-router-dom";
import { ApiError } from "../api";
import { useAuth } from "../auth/AuthContext";
import AuthShell, { ERROR, FIELD, LABEL, SUBMIT } from "../components/AuthShell";

type Mode = "login" | "signup";

const COPY = {
  login: {
    label: "sign in",
    heading: "Welcome back.",
    submit: "Sign in",
    alt: "New here?",
    altLink: "Create an account",
    altTo: "/signup",
  },
  signup: {
    label: "create account",
    heading: "Start reading\nwith citations.",
    submit: "Create account",
    alt: "Already have an account?",
    altLink: "Sign in",
    altTo: "/login",
  },
} as const;

export default function AuthForm({ mode }: { mode: Mode }) {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  // ProtectedRoute stashes the page that sent us here, so an invitation link
  // survives the sign-in it required. Falls back to the app for a plain login.
  const from = (useLocation().state as { from?: Location } | null)?.from;
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const copy = COPY[mode];

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password);
      navigate(from ? `${from.pathname}${from.search}` : "/app", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell label={copy.label} heading={copy.heading}>
      <form onSubmit={submit} className="mt-8 space-y-4">
        <div>
          <label htmlFor="email" className={LABEL}>email</label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={FIELD}
          />
        </div>
        <div>
          <label htmlFor="password" className={LABEL}>password</label>
          <input
            id="password"
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={FIELD}
          />
          {mode === "signup" ? (
            <p className="mt-1.5 font-mono text-[0.6rem] text-faint">at least 8 characters</p>
          ) : (
            // On the sign-in form, not buried after a failed attempt: someone
            // who has forgotten their password already knows it.
            <p className="mt-1.5 text-right">
              <Link
                to="/forgot"
                className="font-mono text-[0.6rem] text-faint underline-offset-2
                           transition-colors hover:text-accent hover:underline"
              >
                forgot your password?
              </Link>
            </p>
          )}
        </div>

        {error && <p role="alert" className={ERROR}>{error}</p>}

        <button type="submit" disabled={busy} className={SUBMIT}>
          {busy ? "…" : copy.submit}
        </button>
      </form>

      <p className="mt-6 font-sans text-sm text-graphite">
        {copy.alt}{" "}
        {/* The destination is forwarded across the login/signup hop too. Someone
            invited by email has no account yet, so this link is exactly the step
            they take — dropping the state here would lose the invitation at the
            last moment, which is when it is hardest to explain. */}
        <Link
          to={copy.altTo}
          state={from ? { from } : undefined}
          className="text-accent underline-offset-2 hover:underline"
        >
          {copy.altLink}
        </Link>
      </p>
    </AuthShell>
  );
}
