/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "Digital manuscript" palette — warm paper, ink, one rubrication accent.
        paper: "#F7F5F0", // page background (warm off-white)
        panel: "#FDFCFA", // subtle raised surfaces / rails
        raised: "#FFFFFF", // fully-raised cards (library, sources, answer)
        ink: "#1C1A17", // primary text (near-black, warm)
        graphite: "#6B675F", // secondary text
        faint: "#A8A399", // tertiary / disabled
        line: "#E4E0D6", // hairline rules
        lineSoft: "#EEEAE0", // softer rule on raised cards
        accent: "#A63A2B", // oxide red — citations, focus, marks
        accentInk: "#8A2E22", // darker accent for hover/press
        accentSoft: "#F3E4E0", // accent wash for highlights
        good: "#5B7A54", // muted green (positive relevance)
      },
      fontFamily: {
        serif: ['"Newsreader"', "Georgia", "serif"],
        // Atkinson Hyperlegible: max character distinction, ideal for a reading tool.
        sans: ['"Atkinson Hyperlegible"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        // Warm, low-opacity elevation scale — paper casts soft umber shadows,
        // never a hard grey. e1 = resting card, e2 = raised/hover, e3 = focus/lift.
        e1: "0 1px 2px rgba(60,42,28,.05), 0 2px 6px rgba(90,64,42,.05)",
        e2: "0 2px 4px rgba(60,42,28,.06), 0 10px 26px rgba(90,64,42,.09)",
        e3: "0 4px 10px rgba(60,42,28,.08), 0 18px 44px rgba(90,64,42,.13)",
      },
    },
  },
  plugins: [],
};
