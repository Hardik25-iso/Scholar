import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { forgotPassword } from "../api";
import AuthShell, { ERROR, FIELD, LABEL, SUBMIT } from "../components/AuthShell";

/**
 * Ask for a reset link.
 *
 * The confirmation deliberately does not say whether the address is registered.
 * The backend answers 202 either way so this route cannot be used to test who
 * has an account here; saying "no such account" in the UI would hand back the
 * exact answer the API refuses to give.
 */
export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await forgotPassword(email);
      setSent(true);
    } catch {
      setError("Could not start a reset just now. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  };

  if (sent)
    return (
      <AuthShell label="check your email" heading={"Sent, if that\naccount exists."}>
        <p className="mt-6 text-sm leading-relaxed text-graphite">
          If <span className="font-mono text-[0.8rem] text-ink">{email}</span> has a
          Scholar account, a reset link is on its way. It works once and expires
          shortly.
        </p>
        <p className="mt-6 font-sans text-sm text-graphite">
          <Link to="/login" className="text-accent underline-offset-2 hover:underline">
            Back to sign in
          </Link>
        </p>
      </AuthShell>
    );

  return (
    <AuthShell label="reset password" heading={"Forgotten\nyour password?"}>
      <p className="mt-4 text-sm leading-relaxed text-graphite">
        Enter the address you signed up with and we will send a link to set a new one.
      </p>
      <form onSubmit={submit} className="mt-6 space-y-4">
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

        {error && <p role="alert" className={ERROR}>{error}</p>}

        <button type="submit" disabled={busy} className={SUBMIT}>
          {busy ? "…" : "Send reset link"}
        </button>
      </form>

      <p className="mt-6 font-sans text-sm text-graphite">
        Remembered it?{" "}
        <Link to="/login" className="text-accent underline-offset-2 hover:underline">
          Sign in
        </Link>
      </p>
    </AuthShell>
  );
}
