import { Link } from "react-router-dom";

interface Props {
  /** Small uppercase kicker above the heading, e.g. "reset password". */
  label: string;
  /** Supports "\n" for a deliberate line break. */
  heading: string;
  children: React.ReactNode;
}

/**
 * The centred card every out-of-app page sits in: sign in, sign up, forgot
 * password, choose a new one, accept an invitation.
 *
 * Extracted once there were four of them. Sharing the frame matters here beyond
 * tidiness — these pages are where someone lands from a link in an email, and a
 * page that does not look like the product is exactly what a phishing page
 * looks like.
 */
export default function AuthShell({ label, heading, children }: Props) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-paper px-6">
      <div className="rise w-full max-w-sm">
        <Link to="/" className="font-serif text-2xl font-semibold tracking-tight text-ink">
          Scholar
        </Link>
        <p className="mt-8 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-faint">
          {label}
        </p>
        <h1 className="mt-2 whitespace-pre-line font-serif text-3xl font-medium leading-tight text-ink">
          {heading}
        </h1>
        {children}
      </div>
    </div>
  );
}

/** Shared input styling — the pages differ in fields, not in how a field looks. */
export const FIELD =
  "mt-1 w-full border border-line bg-panel px-3 py-2.5 font-sans text-sm " +
  "text-ink placeholder:text-faint focus:border-accent focus:outline-none";

export const LABEL =
  "font-mono text-[0.62rem] uppercase tracking-[0.14em] text-graphite";

export const SUBMIT =
  "w-full cursor-pointer bg-ink px-4 py-3 font-mono text-[0.7rem] uppercase " +
  "tracking-[0.16em] text-paper transition-colors duration-200 hover:bg-accent " +
  "disabled:cursor-not-allowed disabled:opacity-40";

export const ERROR =
  "border-l-2 border-accent pl-3 font-mono text-xs leading-relaxed text-accent";
