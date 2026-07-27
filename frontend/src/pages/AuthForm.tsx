import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api";
import { useAuth } from "../auth/AuthContext";

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
      navigate("/app", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-paper px-6">
      <div className="rise w-full max-w-sm">
        <Link to="/" className="font-serif text-2xl font-semibold tracking-tight text-ink">
          Scholar
        </Link>
        <p className="mt-8 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
          {copy.label}
        </p>
        <h1 className="mt-2 whitespace-pre-line font-serif text-3xl font-medium leading-tight text-ink">
          {copy.heading}
        </h1>

        <form onSubmit={submit} className="mt-8 space-y-4">
          <div>
            <label htmlFor="email" className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-graphite">
              email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full border border-line bg-panel px-3 py-2.5 font-sans text-sm
                         text-ink placeholder:text-faint focus:border-accent focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="password" className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-graphite">
              password
            </label>
            <input
              id="password"
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full border border-line bg-panel px-3 py-2.5 font-sans text-sm
                         text-ink placeholder:text-faint focus:border-accent focus:outline-none"
            />
            {mode === "signup" && (
              <p className="mt-1.5 font-mono text-[0.6rem] text-faint">at least 8 characters</p>
            )}
          </div>

          {error && (
            <p role="alert" className="border-l-2 border-accent pl-3 font-mono text-xs leading-relaxed text-accent">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full cursor-pointer bg-ink px-4 py-3 font-mono text-[0.7rem] uppercase
                       tracking-[0.16em] text-paper transition-colors duration-200
                       hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? "…" : copy.submit}
          </button>
        </form>

        <p className="mt-6 font-sans text-sm text-graphite">
          {copy.alt}{" "}
          <Link to={copy.altTo} className="text-accent underline-offset-2 hover:underline">
            {copy.altLink}
          </Link>
        </p>
      </div>
    </div>
  );
}
