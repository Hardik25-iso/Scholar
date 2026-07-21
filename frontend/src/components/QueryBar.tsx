import { useState } from "react";

interface Props {
  disabled: boolean;
  onAsk: (question: string) => void;
}

/** The single input at the bottom of the chat pane. */
export default function QueryBar({ disabled, onAsk }: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    const q = value.trim();
    if (!q || disabled) return;
    onAsk(q);
    setValue("");
  };

  return (
    <div className="border-t border-line bg-paper/90 backdrop-blur px-6 py-4">
      <div className="mx-auto flex max-w-2xl items-center gap-3">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Ask about your papers…"
          disabled={disabled}
          className="flex-1 bg-transparent font-serif text-lg text-ink
                     placeholder:text-faint focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="group flex h-11 w-11 shrink-0 cursor-pointer items-center
                     justify-center rounded-full border border-line text-graphite
                     transition-colors duration-200 hover:border-accent hover:text-accent
                     disabled:cursor-not-allowed disabled:opacity-30
                     disabled:hover:border-line disabled:hover:text-graphite"
          aria-label="Ask"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path
              d="M7 12V2M7 2L2.5 6.5M7 2l4.5 4.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
