/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "Digital manuscript" palette — warm paper, ink, one rubrication accent.
        paper: "#F7F5F0", // page background (warm off-white)
        panel: "#FDFCFA", // raised surfaces (cards)
        ink: "#1C1A17", // primary text (near-black, warm)
        graphite: "#6B675F", // secondary text
        faint: "#A8A399", // tertiary / disabled
        line: "#E4E0D6", // hairline rules
        accent: "#A63A2B", // oxide red — citations, focus, marks
        accentSoft: "#F3E4E0", // accent wash for highlights
      },
      fontFamily: {
        serif: ['"Newsreader"', "Georgia", "serif"],
        // Atkinson Hyperlegible: max character distinction, ideal for a reading tool.
        sans: ['"Atkinson Hyperlegible"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
