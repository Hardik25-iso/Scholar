import { Link } from "react-router-dom";

/** Shared frame for the plain-text pages: data handling, terms. */
export default function LegalPage({
  title, updated, children,
}: { title: string; updated: string; children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <header className="border-b border-line bg-panel/60">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-4">
          <Link to="/" className="font-serif text-xl font-semibold tracking-tight">
            Scholar
          </Link>
          <Link
            to="/"
            className="font-mono text-[0.62rem] uppercase tracking-[0.12em] text-graphite
                       transition-colors hover:text-accent"
          >
            back
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-14">
        <h1 className="font-serif text-4xl font-medium leading-tight">{title}</h1>
        <p className="mt-3 font-mono text-[0.62rem] uppercase tracking-[0.16em] text-faint">
          last updated {updated}
        </p>
        <div className="mt-10">{children}</div>
      </main>
    </div>
  );
}

export function H2({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mt-10 font-serif text-[1.4rem] font-medium leading-snug first:mt-0">
      {children}
    </h2>
  );
}

export function P({ children }: { children: React.ReactNode }) {
  return <p className="mt-3 text-[0.95rem] leading-relaxed text-graphite">{children}</p>;
}

export function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    // Scrolls inside its own box rather than making the page scroll sideways.
    <div className="mt-5 overflow-x-auto">
      <table className="w-full min-w-[34rem] border-collapse text-left text-[0.86rem]">
        <thead>
          <tr>
            {head.map((h) => (
              <th
                key={h}
                className="border-b border-line pb-2 pr-4 font-mono text-[0.6rem]
                           uppercase tracking-[0.14em] text-faint"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row[0]} className="align-top">
              {row.map((cell, i) => (
                <td
                  key={i}
                  className={`border-b border-lineSoft py-3 pr-4 leading-relaxed ${
                    i === 0 ? "font-medium text-ink" : "text-graphite"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
