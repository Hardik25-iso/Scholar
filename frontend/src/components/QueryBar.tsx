import { useState } from "react";
import { SendIcon } from "./icons";

interface Props {
  disabled: boolean;
  onAsk: (question: string) => void;
}

/** The composer at the bottom of the chat pane: an elevated input + send button. */
export default function QueryBar({ disabled, onAsk }: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    const q = value.trim();
    if (!q || disabled) return;
    onAsk(q);
    setValue("");
  };

  return (
    <div className="px-6 pb-5 pt-3">
      <div
        className="mx-auto flex max-w-2xl items-center gap-2 rounded-xl border border-line
                   bg-raised px-2 py-2 pl-4 shadow-e2 transition-all duration-200
                   focus-within:border-accent/60 focus-within:shadow-e3"
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Ask a follow-up about your papers…"
          disabled={disabled}
          className="flex-1 bg-transparent font-sans text-[0.95rem] text-ink
                     placeholder:text-faint focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center
                     rounded-lg bg-accent text-paper shadow-e1 transition-all duration-150
                     hover:-translate-y-px hover:bg-accentInk
                     disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:translate-y-0 disabled:hover:bg-accent"
          aria-label="Ask"
        >
          <SendIcon className="h-[1.05rem] w-[1.05rem]" />
        </button>
      </div>
      <p className="mx-auto mt-2 max-w-2xl text-center font-mono text-[0.6rem] tracking-[0.04em] text-faint">
        Answers use only your indexed papers, cited to the page.
      </p>
    </div>
  );
}
